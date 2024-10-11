from typing import Callable

import numpy as np
import pandas as pd
import reciprocalspaceship as rs

from .tv import TvDenoiseResult, tv_denoise_difference_map
from .utils import (
    canonicalize_amplitudes,
    complex_array_to_rs_dataseries,
    compute_coefficients_from_map,
    compute_map_from_coefficients,
    resolution_limits,
    rs_dataseies_to_complex_array,
)


def _l1_norm(
    array1: np.ndarray,
    array2: np.ndarray,
) -> float:
    assert array1.shape == array2.shape
    return np.sum(np.abs(array1 - array2)) / float(np.prod(array1.shape))


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
    tv_denoise_function: Callable
        A function capable of applying the TV denoising operation to *Fourier space* objects. This
        function should therefore map one complex np.ndarray to a denoised complex np.ndarray, with
        no additional parameters.
    convergance_tolerance: float
        If the change in the estimated derivative SFs drops below this value (L1, per-component)
        then return
    max_iterations: int
        If this number of iterations is reached, stop early.

    Returns
    -------
    estimated_complex_derivative: np.ndarray
        The derivative SFs, with the same amplitudes but phases altered to minimize the TV.
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

        change = _l1_norm(derivative, updated_derivative)
        derivative = updated_derivative
        difference = derivative - native

        converged = change < convergence_tolerance
        num_iterations += 1

        metadata.append(
            {
                "iteration": num_iterations,
                "tv_weight": tv_metadata.optimal_lambda,
                "negentropy": tv_metadata.optimal_negentropy,
                "change": change,
            }
        )

        # TODO remove
        print(num_iterations, tv_metadata.optimal_lambda, tv_metadata.optimal_negentropy, change)

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
) -> rs.DataSet:
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
    """

    # TODO think about this -- better way to enumerate?
    # fills in missing reflections

    m = compute_map_from_coefficients(
        map_coefficients=input_dataset,
        amplitude_label=native_amplitude_column,
        phase_label=calculated_phase_column,
        map_sampling=3,
    )
    _, hrl = resolution_limits(input_dataset)
    d = compute_coefficients_from_map(
        ccp4_map=m,
        high_resolution_limit=hrl,
        amplitude_label=native_amplitude_column,
        phase_label=calculated_phase_column,
    )

    missing_indices = d.index.difference(input_dataset.index)
    df_missing = rs.DataSet(0.0, index=missing_indices, columns=input_dataset.columns)
    df_missing.spacegroup = input_dataset.spacegroup
    df_missing.cell = input_dataset.cell
    input_dataset = rs.concat([input_dataset, df_missing])

    input_dataset[native_amplitude_column] = input_dataset[native_amplitude_column].astype(
        rs.StructureFactorAmplitudeDtype()
    )
    input_dataset[derivative_amplitude_column] = input_dataset[derivative_amplitude_column].astype(
        rs.StructureFactorAmplitudeDtype()
    )
    input_dataset[calculated_phase_column] = input_dataset[calculated_phase_column].astype(
        rs.PhaseDtype()
    )

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # TODO we could swap from this closure to a class
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

    output_dataset = input_dataset.copy()
    output_dataset[output_derivative_phase_column] = derivative_phases.astype(rs.PhaseDtype())

    # TODO remove block below
    difference_amplitudes, difference_phases = complex_array_to_rs_dataseries(
        it_tv_complex_derivative - native, input_dataset.index
    )
    output_dataset["DF"] = difference_amplitudes.astype(rs.StructureFactorAmplitudeDtype())
    output_dataset["DPHI"] = difference_phases.astype(rs.PhaseDtype())
    # ^^^^^^^^^^^^^^^^^^^^^^^

    canonicalize_amplitudes(
        output_dataset,
        amplitude_label=derivative_amplitude_column,
        phase_label=output_derivative_phase_column,
        inplace=True,
    )

    return output_dataset, metadata
