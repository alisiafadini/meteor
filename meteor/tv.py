import numpy as np
import gemmi
import reciprocalspaceship as rs

from skimage.restoration import denoise_tv_chambolle
from scipy.optimize import minimize_scalar

from typing import Sequence

from .validate import negentropy
from .utils import (
    compute_map_from_coefficients,
    compute_coefficients_from_map,
    resolution_limits,
    numpy_array_to_map,
)
from .settings import (
    TV_LAMBDA_RANGE,
    TV_STOP_TOLERANCE,
    TV_MAP_SAMPLING,
    TV_MAX_NUM_ITER,
)


def _tv_denoise_array(*, map_as_array: np.ndarray, weight: float) -> np.ndarray:
    """
    Closure convienence function to generate more readable code.
    """
    denoised_map = denoise_tv_chambolle(
        map_as_array,
        weight=weight,
        eps=TV_STOP_TOLERANCE,
        max_num_iter=TV_MAX_NUM_ITER,
    )
    return denoised_map


def tv_denoise_difference_map(
    difference_map_coefficients: rs.DataSet,
    difference_map_amplitude_column: str = "DF",
    difference_map_phase_column: str = "PHIC",
    lambda_values_to_scan: Sequence[float] | None = None,
) -> tuple[rs.DataSet, float]:
    """ lambda_values_to_scan = None --> Golden method """

    # TODO write decent docstring

    difference_map = compute_map_from_coefficients(
        map_coefficients=difference_map_coefficients,
        amplitude_label=difference_map_amplitude_column,
        phase_label=difference_map_phase_column,
        map_sampling=TV_MAP_SAMPLING,
    )

    difference_map_as_array = np.array(difference_map.grid)

    def negentropy_objective(tv_lambda: float):
        denoised_map = _tv_denoise_array(
            map_as_array=difference_map_as_array, weight=tv_lambda
        )
        return negentropy(denoised_map.flatten())

    optimal_lambda: float
    if lambda_values_to_scan is not None:
        highest_negentropy = -1e8
        for tv_lambda in lambda_values_to_scan:
            trial_negentropy = negentropy_objective(tv_lambda)
            if negentropy_objective(tv_lambda) > highest_negentropy:
                optimal_lambda = tv_lambda
                highest_negentropy = trial_negentropy
    else:
        optimizer_result = minimize_scalar(
            negentropy_objective, bracket=TV_LAMBDA_RANGE, method="golden"
        )
        assert (
            optimizer_result.success
        ), "Golden minimization failed to find optimal TV lambda"
        optimal_lambda = optimizer_result.x

    final_map = _tv_denoise_array(
        map_as_array=difference_map_as_array, weight=optimal_lambda
    )
    final_map = numpy_array_to_map(
        final_map,
        spacegroup=difference_map_coefficients.spacegroup,
        cell=difference_map_coefficients.cell,
    )

    _, dmin = resolution_limits(difference_map_coefficients)
    final_map_coefficients = compute_coefficients_from_map(
        ccp4_map=final_map,
        high_resolution_limit=dmin,
        amplitude_label=difference_map_amplitude_column,
        phase_label=difference_map_phase_column,
    )

    return final_map_coefficients


def iterative_tv_phase_retrieval(): ...
