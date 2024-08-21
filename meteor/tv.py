import numpy as np
import gemmi as gm
import reciprocalspaceship as rs

from skimage.restoration import denoise_tv_chambolle
from scipy.optimize import minimize_scalar

from typing import Sequence

from .validate import negentropy
from .utils import compute_map_from_coefficients, compute_coefficients_from_map, resolution_limits
from .settings import (
    TV_LAMBDA_RANGE,
    TV_STOP_TOLERANCE,
    TV_MAP_SAMPLING,
    TV_MAX_NUM_ITER,
    TV_AMPLITUDE_LABEL,
    TV_PHASE_LABEL,
)


def _tv_denoise_ccp4_map(*, map: gm.Ccp4Map, weight: float) -> np.ndarray:
    """
    Closure convienence function to generate more readable code.
    """
    denoised_map = denoise_tv_chambolle(
        np.array(map.grid),
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
    """

    lambda_values_to_scan = None --> Golden method

    Returns:
       rs.Dataset: denoised dataset with new columns `DFtv`, `DPHItv`
    """

    # TODO write decent docstring

    difference_map = compute_map_from_coefficients(
        map_coefficients=difference_map_coefficients,
        amplitude_label=difference_map_amplitude_column,
        phase_label=difference_map_phase_column,
        map_sampling=TV_MAP_SAMPLING,
    )

    def negentropy_objective(tv_lambda: float):
        denoised_map = _tv_denoise_ccp4_map(map=difference_map, weight=tv_lambda)
        return negentropy(denoised_map.flatten())
    
    optimal_lambda: float

    if lambda_values_to_scan:
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
        assert optimizer_result.success, "Golden minimization failed to find optimal TV lambda"
        optimal_lambda = optimizer_result.x

    final_map_array = _tv_denoise_ccp4_map(map=difference_map, weight=optimal_lambda)

    # TODO: verify correctness

    _, high_resolution_limit = resolution_limits(difference_map_coefficients)
    final_map_coefficients = compute_coefficients_from_map(
        map=final_map_array,
        high_resolution_limit=high_resolution_limit,
        amplitude_label=TV_AMPLITUDE_LABEL,
        phase_label=TV_PHASE_LABEL,
    )

    # TODO: need to be sure HKLs line up
    difference_map_coefficients[[TV_AMPLITUDE_LABEL]] = np.abs(final_map_coefficients)
    difference_map_coefficients[[TV_PHASE_LABEL]] = np.angle(final_map_coefficients, deg=True)

    return difference_map_coefficients


def iterative_tv_phase_retrieval(): ...
