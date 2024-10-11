from typing import Callable

import numpy as np
import pandas as pd
import reciprocalspaceship as rs

from .tv import TvDenoiseResult, tv_denoise_difference_map
from .utils import (
    average_phase_diff_in_degrees,
    canonicalize_amplitudes,
    complex_array_to_rs_dataseries,
    rs_dataseies_to_complex_array,
)


def _project_derivative_on_experimental_set(
    *,
    native: np.ndarray,
    derivative_amplitudes: np.ndarray,
    difference: np.ndarray,
) -> np.ndarray:
    """
    Project `derivative` onto the set of experimentally observed amplitudes,

        Fh' = (D_F' + F) * [|Fh| / |D_F' + F|]

    In English, the output is a complex-valued array that changes the derivative phase to ensure:

        difference = derivative - native

    Parameters
    ----------
    native: np.ndarray
        The experimentally observed native amplitudes and computed phases, as a complex array.
    derivative_amplitudes: np.ndarray
        An array of the experimentally observed derivative amplitudes. Typically real-valued, but
        a complex-valued array with arbitrary phase can be passed (phases discarded).
    difference: np.ndarray
        The estimated complex structure factor difference, derivative-minus-native.

    Returns
    -------
    projected_derivative: np.ndarray
        The complex-valued derivative structure factors, with experimental amplitude and phase
        adjusted to ensure that difference = derivative - native.
    """
    projected_derivative = difference + native
    projected_derivative *= np.abs(derivative_amplitudes) / np.abs(projected_derivative)
    return projected_derivative


def _complex_derivative_from_iterative_tv(
    *,
    native: np.ndarray,
    initial_derivative: np.ndarray,
    tv_denoise_function: Callable[[np.ndarray], tuple[np.ndarray, TvDenoiseResult]],
    convergence_tolerance: float = 1e-4,
    max_iterations: int = 1000,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Estimate the derivative phases using the iterative TV algorithm.

    This function contains the algorithm logic.

    Parameters
    ----------
    complex_native: np.ndarray
        The complex native structure factors, usually experimental amplitudes and calculated phases

    initial_complex_derivative : np.ndarray
        The complex derivative structure factors, usually with experimental amplitudes and esimated
        phases (often calculated from the native structure)

    tv_denoise_function: Callable[[np.ndarray], tuple[np.ndarray, TvDenoiseResult]]
        A function capable of applying the TV denoising operation to *Fourier space* objects. This
        function should therefore map one complex np.ndarray to a denoised complex np.ndarray and
        the TvDenoiseResult for that TV run.

    convergance_tolerance: float
        If the change in the estimated derivative SFs drops below this value (phase, per-component)
        then return

    max_iterations: int
        If this number of iterations is reached, stop early.

    Returns
    -------
    estimated_complex_derivative: np.ndarray
        The derivative SFs, with the same amplitudes but phases altered to minimize the TV.

    metadata: pd.DataFrame
        Information about the algorithm run as a function of iteration. For each step, includes:
        the tv_weight used, the negentropy (after the TV step), and the average phase change in
        degrees.
    """

    derivative = np.copy(initial_derivative)
    difference = initial_derivative - native

    converged: bool = False
    num_iterations: int = 0
    metadata: list[dict[str, float]] = []

    while not converged:
        difference_tvd, tv_metadata = tv_denoise_function(difference)
        updated_derivative = _project_derivative_on_experimental_set(
            native=native,
            derivative_amplitudes=np.abs(derivative),
            difference=difference_tvd,
        )

        phase_change = average_phase_diff_in_degrees(derivative, updated_derivative)
        derivative = updated_derivative
        difference = derivative - native

        converged = phase_change < convergence_tolerance
        num_iterations += 1

        metadata.append(
            {
                "iteration": num_iterations,
                "tv_weight": tv_metadata.optimal_lambda,
                "negentropy": tv_metadata.optimal_negentropy,
                "average_phase_change": phase_change,
            }
        )

        if num_iterations > max_iterations:
            break

    return derivative, pd.DataFrame(metadata)


def iterative_tv_phase_retrieval(
    input_dataset: rs.DataSet,
    *,
    native_amplitude_column: str = "F",
    derivative_amplitude_column: str = "Fh",
    calculated_phase_column: str = "PHIC",
    output_derivative_phase_column: str = "PHICh",
    convergence_tolerance: float = 1e-3,
    max_iterations: int = 100,
    tv_weights_to_scan: list[float] = [0.001, 0.01, 0.1, 1.0],
) -> tuple[rs.DataSet, pd.DataFrame]:
    """
    Here is a brief psuedocode sketch of the alogrithm. Structure factors F below are complex unless
    explicitly annotated |*|.

        Input: |F|, |Fh|, phi_c
        Note: F = |F| * exp{ phi_c } is the native/dark data,
             |Fh| represents the derivative/triggered/light data

        Initialize:
         - D_F = ( |Fh| - |F| ) * exp{ phi_c }

        while not converged:
            D_rho = FT{ D_F }                       Fourier transform
            D_rho' = TV{ D_rho }                    TV denoise: apply real space prior
            D_F' = FT-1{ D_rho' }                   back Fourier transform
            Fh' = (D_F' + F) * [|Fh| / |D_F' + F|]  Fourier space projection onto experimental set
            D_F = Fh' - F

    Where the TV lambda parameter is determined using golden section optimization. The algorithm
    iterates until the changes in DF for each iteration drop below a specified per-amplitude
    threshold.

    Parameters
    ----------
    input_dataset : rs.DataSet
        The input dataset containing the native and derivative amplitude columns, as well as
        the calculated phase column.

    native_amplitude_column : str, optional
        Column name in `input_dataset` representing the amplitudes of the native (dark) structure
        factors, by default "F".

    derivative_amplitude_column : str, optional
        Column name in `input_dataset` representing the amplitudes of the derivative (light)
        structure factors, by default "Fh".

    calculated_phase_column : str, optional
        Column name in `input_dataset` representing the phases of the native (dark) structure
        factors, by default "PHIC".

    output_derivative_phase_column : str, optional
        Column name where the estimated derivative phases will be stored in the output dataset,
        by default "PHICh".

    convergance_tolerance: float
        If the change in the estimated derivative SFs drops below this value (phase, per-component)
        then return

    max_iterations: int
        If this number of iterations is reached, stop early.

    tv_weights_to_scan : list[float], optional
        A list of TV regularization weights (λ values) to be scanned for optimal results,
        by default [0.001, 0.01, 0.1, 1.0].

    Returns
    -------
    output_dataset: rs.DataSet
        The estimated derivative phases, along with the input amplitudes and input computed phases.

    metadata: pd.DataFrame
        Information about the algorithm run as a function of iteration. For each step, includes:
        the tv_weight used, the negentropy (after the TV step), and the average phase change in
        degrees.
    """

    # clean TV denoising interface that is crystallographically intelligent
    # maintains state for the HKL index, spacegroup, and cell information
    def tv_denoise_closure(difference: np.ndarray) -> tuple[np.ndarray, TvDenoiseResult]:
        delta_amp, delta_phase = complex_array_to_rs_dataseries(
            difference, index=input_dataset.index
        )

        # these two names are only used inside this closure
        delta_amp.name = "DF_for_tv_closure"
        delta_phase.name = "DPHI_for_tv_closure"

        diffmap = rs.concat([delta_amp, delta_phase], axis=1)
        diffmap.cell = input_dataset.cell
        diffmap.spacegroup = input_dataset.spacegroup

        denoised_map_coefficients, tv_metadata = tv_denoise_difference_map(
            diffmap,
            difference_map_amplitude_column=delta_amp.name,
            difference_map_phase_column=delta_phase.name,
            lambda_values_to_scan=tv_weights_to_scan,
            full_output=True,
        )

        denoised_difference = rs_dataseies_to_complex_array(
            denoised_map_coefficients[delta_amp.name], denoised_map_coefficients[delta_phase.name]
        )

        return denoised_difference, tv_metadata

    # convert the native and derivative datasets to complex arrays
    native = rs_dataseies_to_complex_array(
        input_dataset[native_amplitude_column], input_dataset[calculated_phase_column]
    )
    initial_derivative = rs_dataseies_to_complex_array(
        input_dataset[derivative_amplitude_column], input_dataset[calculated_phase_column]
    )

    # estimate the derivative phases using the iterative TV algorithm
    it_tv_complex_derivative, metadata = _complex_derivative_from_iterative_tv(
        native=native,
        initial_derivative=initial_derivative,
        tv_denoise_function=tv_denoise_closure,
        convergence_tolerance=convergence_tolerance,
        max_iterations=max_iterations,
    )
    _, derivative_phases = complex_array_to_rs_dataseries(
        it_tv_complex_derivative, input_dataset.index
    )

    # combine the determined derivative phases with the input to generate a complete output
    output_dataset = input_dataset.copy()
    output_dataset[output_derivative_phase_column] = derivative_phases.astype(rs.PhaseDtype())
    canonicalize_amplitudes(
        output_dataset,
        amplitude_label=derivative_amplitude_column,
        phase_label=output_derivative_phase_column,
        inplace=True,
    )

    return output_dataset, metadata