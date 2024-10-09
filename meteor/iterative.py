from typing import Callable

import numpy as np
import pandas as pd
import reciprocalspaceship as rs

from .tv import tv_denoise_difference_map
from .utils import canonicalize_amplitudes


def _rs_dataseies_to_complex_array(
    amplitudes: rs.DataSeries,
    phases: rs.DataSeries,
) -> np.ndarray:
    pd.testing.assert_index_equal(amplitudes.index, phases.index)
    return amplitudes.to_numpy() * np.exp(1j * np.deg2rad(phases.to_numpy()))


def _complex_array_to_rs_dataseries(
    complex_array: np.ndarray,
    index: pd.Index,
) -> tuple[rs.DataSeries, rs.DataSeries]:
    amplitudes = rs.DataSeries(np.abs(complex_array), index=index)
    amplitudes = amplitudes.astype(rs.StructureFactorAmplitudeDtype())
    phases = rs.DataSeries(np.rad2deg(np.angle(complex_array)), index=index)
    phases = phases.astype(rs.PhaseDtype())
    return amplitudes, phases


def _l1_norm(
    array1: np.ndarray,
    array2: np.ndarray,
) -> float:
    assert array1.shape == array2.shape
    return np.sum(np.abs(array1 - array2)) / float(np.prod(array1.shape))


def _project_derivative_on_experimental_set(
    *,
    native: np.ndarray,
    derivative: np.ndarray,
    difference: np.ndarray,
) -> np.ndarray:
    # TODO docstring
    proj_derivative = difference + native
    proj_derivative /= np.abs(derivative) / np.abs(proj_derivative)
    return proj_derivative


def _complex_derivative_from_iterative_tv(
    *,
    complex_native: np.ndarray,
    initial_complex_derivative: np.ndarray,
    tv_denoise_closure: Callable[[np.ndarray], np.ndarray],  # TODO callable
    convergence_tolerance: float = 0.01,
) -> np.ndarray:
    # TODO docstring

    for input_array in [complex_native, initial_complex_derivative]:
        assert input_array.dtype is np.complex128

    complex_difference = initial_complex_derivative - complex_native
    converged: bool = False
    complex_derivative = np.copy(initial_complex_derivative)

    while not converged:
        complex_difference_tvd = tv_denoise_closure(complex_difference)
        updated_complex_derivative = _project_derivative_on_experimental_set(
            native=complex_native,
            derivative=complex_derivative,
            difference=complex_difference_tvd,
        )

        change = _l1_norm(complex_derivative, updated_complex_derivative)
        complex_derivative = updated_complex_derivative
        complex_difference = complex_derivative - complex_native

        converged = change < convergence_tolerance

    return complex_derivative


def iterative_tv_phase_retrieval(
    input_dataset: rs.DataSet,
    *,
    native_amplitude_column: str = "F",
    derivative_amplitude_column: str = "Fh",
    calculated_phase_column: str = "PHIC",
    output_derivative_phase_column: str = "PHICh",
    convergence_tolerance: float = 0.01,
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

    complex_native = _rs_dataseies_to_complex_array(
        input_dataset[native_amplitude_column], input_dataset[calculated_phase_column]
    )
    initial_complex_derivative = _rs_dataseies_to_complex_array(
        input_dataset[derivative_amplitude_column], input_dataset[calculated_phase_column]
    )

    def tv_denoise_closure(complex_difference: np.ndarray) -> np.ndarray:
        delta_amp, delta_phase = _complex_array_to_rs_dataseries(
            complex_difference, index=input_dataset.index
        )
        delta_amp.name = "DF"
        delta_phase.name = "PHIC"
        diffmap = rs.concat([delta_amp, delta_phase])
        diffmap.cell = input_dataset.cell
        diffmap.spacegroup = input_dataset.spacegroup
        return tv_denoise_difference_map(
            diffmap,
            lambda_values_to_scan=[
                0.01,
            ],  # TODO
        )

    updated_initial_complex_derivative = _complex_derivative_from_iterative_tv(
        complex_native=complex_native,
        initial_complex_derivative=initial_complex_derivative,
        tv_denoise_closure=tv_denoise_closure,
        convergence_tolerance=convergence_tolerance,
    )

    derivative_amplitudes, derivative_phases = _complex_array_to_rs_dataseries(
        updated_initial_complex_derivative, input_dataset.index
    )

    output_dataset = input_dataset.copy()
    output_dataset[output_derivative_phase_column] = derivative_phases

    # TODO, probably not needed
    pd.testing.assert_series_equal(
        input_dataset[derivative_amplitude_column], derivative_amplitudes
    )

    canonicalize_amplitudes(
        output_dataset,
        amplitude_label=derivative_amplitude_column,
        phase_label=output_derivative_phase_column,
        inplace=True,
    )

    return output_dataset
