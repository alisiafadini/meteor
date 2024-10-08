from dataclasses import dataclass, field
from typing import Literal, Sequence, overload

import numpy as np
import reciprocalspaceship as rs
from skimage.restoration import denoise_tv_chambolle

from .settings import (
    TV_LAMBDA_RANGE,
    TV_MAP_SAMPLING,
    TV_MAX_NUM_ITER,
    TV_STOP_TOLERANCE,
)
from .utils import (
    canonicalize_amplitudes,
    compute_coefficients_from_map,
    compute_map_from_coefficients,
    numpy_array_to_map,
    resolution_limits,
)
from .validate import ScalarMaximizer, negentropy


@dataclass
class TvDenoiseResult:
    optimal_lambda: float
    optimal_negentropy: float
    map_sampling_used_for_tv: float
    lambdas_scanned: set[float] = field(default_factory=set)


def _tv_denoise_array(*, map_as_array: np.ndarray, weight: float) -> np.ndarray:
    """Closure convienence function to generate more readable code."""
    denoised_map = denoise_tv_chambolle(
        map_as_array,
        weight=weight,
        eps=TV_STOP_TOLERANCE,
        max_num_iter=TV_MAX_NUM_ITER,
    )
    return denoised_map


@overload
def tv_denoise_difference_map(
    difference_map_coefficients: rs.DataSet,
    full_output: Literal[False] = False,
    difference_map_amplitude_column: str = "DF",
    difference_map_phase_column: str = "PHIC",
    lambda_values_to_scan: Sequence[float] | np.ndarray | None = None,
) -> rs.DataSet: ...


@overload
def tv_denoise_difference_map(
    difference_map_coefficients: rs.DataSet,
    full_output: Literal[True],
    difference_map_amplitude_column: str = "DF",
    difference_map_phase_column: str = "PHIC",
    lambda_values_to_scan: Sequence[float] | np.ndarray | None = None,
) -> tuple[rs.DataSet, TvDenoiseResult]: ...


def tv_denoise_difference_map(
    difference_map_coefficients: rs.DataSet,
    full_output: bool = False,
    difference_map_amplitude_column: str = "DF",
    difference_map_phase_column: str = "PHIC",
    lambda_values_to_scan: Sequence[float] | np.ndarray | None = None,
) -> rs.DataSet | tuple[rs.DataSet, TvDenoiseResult]:
    """Single-pass TV denoising of a difference map.

    Automatically selects the optimal level of regularization (the TV lambda parameter) by
    maximizing the negentropy of the denoised map. Two modes can be used to dictate which
    candidate values of lambda are assessed:

      1. By default (`lambda_values_to_scan=None`), the golden-section search algorithm selects
         a lambda value according to the bounds and convergence criteria set in meteor.settings.
      2. Alternatively, an explicit list of lambda values to assess can be provided using
        `lambda_values_to_scan`.


    Parameters
    ----------
    difference_map_coefficients : rs.DataSet
        The input dataset containing the difference map coefficients (amplitude and phase)
        that will be used to compute the difference map.
    full_output : bool, optional
        If `True`, the function returns both the denoised map coefficients and a `TvDenoiseResult`
         object containing the optimal lambda and the associated negentropy. If `False`, only
         the denoised map coefficients are returned. Default is `False`.
    difference_map_amplitude_column : str, optional
        The column name in `difference_map_coefficients` that contains the amplitude values for
        the difference map. Default is "DF".
    difference_map_phase_column : str, optional
        The column name in `difference_map_coefficients` that contains the phase values for the
        difference map. Default is "PHIC".
    lambda_values_to_scan : Sequence[float] | None, optional
        A sequence of lambda values to explicitly scan for determining the optimal value. If
        `None`, the function uses the golden-section search method to determine the optimal
        lambda. Default is `None`.

    Returns
    -------
    rs.DataSet | tuple[rs.DataSet, TvDenoiseResult]
        If `full_output` is `False`, returns a `rs.DataSet`, which is a new DataSet with the
          denoised difference map amplitudes and phases in two columns, named
          `difference_map_amplitude_column` and `difference_map_phase_column` respectively.
        If `full_output` is `True`, returns a tuple containing:
        - `rs.DataSet`: The denoised map coefficients.
        - `TvDenoiseResult`: An object w/ the optimal lambda and the corresponding negentropy.

    Raises
    ------
    AssertionError
        If the golden-section search fails to find an optimal lambda.

    Notes
    -----
    - The function is designed to maximize the negentropy of the denoised map, which is a
      measure of the map's "randomness."
      Higher negentropy generally corresponds to a more informative and less noisy map.
    - The golden-section search is a robust method for optimizing unimodal functions,
      particularly suited for scenarios where an explicit list of candidate values is not provided.

    Example
    -------
    >>> coefficients = rs.read_mtz("./path/to/difference_map.mtz")  # load dataset
    >>> denoised_map, result = tv_denoise_difference_map(coefficients, full_output=True)
    >>> print(f"Optimal Lambda: {result.optimal_lambda}, Negentropy: {result.optimal_negentropy}")

    """
    difference_map = compute_map_from_coefficients(
        map_coefficients=difference_map_coefficients,
        amplitude_label=difference_map_amplitude_column,
        phase_label=difference_map_phase_column,
        map_sampling=TV_MAP_SAMPLING,
    )
    difference_map_as_array = np.array(difference_map.grid)

    def negentropy_objective(tv_lambda: float):
        denoised_map = _tv_denoise_array(map_as_array=difference_map_as_array, weight=tv_lambda)
        return negentropy(denoised_map.flatten())

    maximizer = ScalarMaximizer(objective=negentropy_objective)
    if lambda_values_to_scan is not None:
        maximizer.optimize_over_explicit_values(arguments_to_scan=lambda_values_to_scan)
    else:
        maximizer.optimize_with_golden_algorithm(bracket=TV_LAMBDA_RANGE)

    # denoise using the optimized parameters and convert to an rs.DataSet
    final_map = _tv_denoise_array(
        map_as_array=difference_map_as_array, weight=maximizer.argument_optimum
    )
    final_map_as_ccp4 = numpy_array_to_map(
        final_map,
        spacegroup=difference_map_coefficients.spacegroup,
        cell=difference_map_coefficients.cell,
    )
    _, dmin = resolution_limits(difference_map_coefficients)
    final_map_coefficients = compute_coefficients_from_map(
        ccp4_map=final_map_as_ccp4,
        high_resolution_limit=dmin,
        amplitude_label=difference_map_amplitude_column,
        phase_label=difference_map_phase_column,
    )

    if full_output:
        tv_result = TvDenoiseResult(
            optimal_lambda=maximizer.argument_optimum,
            optimal_negentropy=maximizer.objective_maximum,
            map_sampling_used_for_tv=TV_MAP_SAMPLING,
            lambdas_scanned=maximizer.values_evaluated,
        )
        return final_map_coefficients, tv_result
    else:
        return final_map_coefficients


def _dataseries_l1_norm(
    series1: rs.DataSeries,
    series2: rs.DataSeries,
) -> float:
    difference = (series2 - series1).dropna()
    num_datapoints = len(difference)
    if num_datapoints == 0:
        raise RuntimeError("no overlapping indices between `series1` and `series2`")
    return np.sum(np.abs(difference)) / float(num_datapoints)


def _projected_derivative_phase(
    *,
    difference_amplitudes: rs.DataSeries,
    difference_phases: rs.DataSeries,
    native_amplitudes: rs.DataSeries,
    native_phases: rs.DataSeries,
) -> rs.DataSeries:
    complex_difference = difference_amplitudes * np.exp(difference_phases)
    complex_native = native_amplitudes * np.exp(native_phases)
    complex_derivative_estimate = complex_difference + complex_native
    return complex_derivative_estimate.apply(np.angle).apply(np.rad2deg)


def iterative_tv_phase_retrieval(
    *,
    input_dataset: rs.DataSet,
    native_amplitude_column: str = "F",
    derivative_amplitude_column: str = "FH",
    calculated_phase_column: str = "PHIC",
    output_derivative_phase_column: str = "PHICH",
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

    # TODO should these be adjustable input params?
    difference_amplitude_column: str = "DF"
    difference_phase_column: str = "DPHIC"

    initial_derivative_phases = (
        input_dataset[calculated_phase_column].copy().rename(output_derivative_phase_column)
    )

    initial_difference_amplitudes = (
        (
            input_dataset[derivative_amplitude_column].copy()
            - input_dataset[native_amplitude_column].copy()
        ).rename(difference_amplitude_column),
    )
    initial_difference_phases = (
        input_dataset[calculated_phase_column].copy().rename(difference_phase_column)
    )

    # working_ds holds 3x complex numbers: native, derivative, differences
    working_ds = rs.concat(
        [
            input_dataset[
                [native_amplitude_column, derivative_amplitude_column, calculated_phase_column]
            ].copy(),
            initial_derivative_phases,
            initial_difference_amplitudes,
            initial_difference_phases,
        ],
        axis=1,
    )

    # begin iterative TV algorithm
    converged: bool = False

    while not converged:
        # TV denoise using golden algorithm, includes forward and backwards FTs
        # this is the un-projected difference amplitude and phase
        # > returned DF_prime is a copy with new `DIFFERENCE_AMPLITUDES` & `calculated_phase_column`
        DF_prime = tv_denoise_difference_map(  # noqa: N806
            difference_map_coefficients=working_ds,
            difference_map_amplitude_column=difference_amplitude_column,
            difference_map_phase_column=difference_phase_column,
        )

        change_in_DF = _dataseries_l1_norm(  # noqa: N806
            working_ds[difference_amplitude_column],  # previous iteration
            DF_prime[difference_amplitude_column],  # current iteration
        )
        converged = change_in_DF < convergence_tolerance

        # update working_ds, NB native and derivative amplitudes & native phases stay the same
        working_ds[output_derivative_phase_column] = _projected_derivative_phase(
            difference_amplitudes=DF_prime[difference_amplitude_column],
            difference_phases=DF_prime[difference_phase_column],
            native_amplitudes=working_ds[native_amplitude_column],
            native_phases=working_ds[calculated_phase_column],
        )

        current_complex_native = working_ds[native_amplitude_column] * np.exp(
            working_ds[calculated_phase_column]
        )
        current_complex_derivative = working_ds[derivative_amplitude_column] * np.exp(
            working_ds[output_derivative_phase_column]
        )
        current_complex_difference = current_complex_derivative - current_complex_native

        working_ds[difference_amplitude_column] = np.abs(current_complex_difference)
        working_ds[difference_phase_column] = np.rad2deg(np.angle(current_complex_difference))

    canonicalize_amplitudes(
        working_ds,
        amplitude_label=derivative_amplitude_column,
        phase_label=output_derivative_phase_column,
        inplace=True,
    )
    canonicalize_amplitudes(
        working_ds,
        amplitude_label=difference_amplitude_column,
        phase_label=difference_phase_column,
        inplace=True,
    )

    return working_ds
