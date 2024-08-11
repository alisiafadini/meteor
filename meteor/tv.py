import numpy as np
import gemmi as gm
import reciprocalspaceship as rs

from skimage.restoration import denoise_tv_chambolle
from scipy.optimize import minimize_scalar

from .utils import compute_map_from_coefficients, compute_coefficients_from_map
from .settings import (
    TV_LAMBDA_RANGE,
    TV_STOP_TOLERANCE,
    TV_MAP_SAMPLING,
    TV_MAX_NUM_ITER,
    TV_AMPLITUDE_LABEL,
    TV_PHASE_LABEL,
)


def tv_denoise_difference_map(
    difference_map_coefficients: rs.DataSet,
    difference_map_amplitude_column: str = "DF",
    difference_map_phase_column: str = "PHIC",
) -> tuple[rs.DataSet, float]:
    """
    Returns:
       rs.Dataset: denoised dataset with new columns `DFtv`, `DPHItv`
    """

    difference_map = compute_map_from_coefficients(
        map_coefficients=difference_map_coefficients,
        amplitude_label=difference_map_amplitude_column,
        phase_label=difference_map_phase_column,
        map_sampling=TV_MAP_SAMPLING,
    )

    def negentropy_objective(tv_lambda: float):
        return denoise_tv_chambolle(
            np.array(difference_map.grid),
            eps=TV_STOP_TOLERANCE,
            weight=tv_lambda,
            max_num_iter=TV_MAX_NUM_ITER,
        )

    optimizer_result = minimize_scalar(
        negentropy_objective, bracket=TV_LAMBDA_RANGE, method="golden"
    )
    assert optimizer_result.success

    optimal_lambda: float = optimizer_result.x

    final_map_array: np.ndarray = denoise_tv_chambolle(
        np.array(difference_map.grid),
        eps=TV_STOP_TOLERANCE,
        weight=optimal_lambda,
        max_num_iter=TV_MAX_NUM_ITER,
    )

    # TODO: we may be able to simplify the code by going directly from a numpy
    #       array to rs.DataSet here -- right now, we go through gemmi format

    ccp4_map = gm.Ccp4Map()

    ccp4_map.grid = gm.FloatGrid(final_map_array)
    ccp4_map.grid.set_unit_cell(gm.UnitCell(*difference_map_coefficients.cell))
    ccp4_map.grid.set_size(difference_map_coefficients.get_reciprocal_grid_size())
    ccp4_map.grid.spacegroup = gm.find_spacegroup_by_name(
        difference_map_coefficients.space_group
    )
    ccp4_map.grid.symmetrize_max()
    ccp4_map.update_ccp4_header()

    high_resolution_limit = np.min(difference_map_coefficients.compute_dHKL())
    denoised_dataset = compute_coefficients_from_map(
        ccp4_map=ccp4_map,
        high_resolution_limit=high_resolution_limit,
        amplitude_label=TV_AMPLITUDE_LABEL,
        phase_label=TV_PHASE_LABEL,
    )

    # ^^^ replace this with something better!

    return denoised_dataset


def iterative_tv_phase_retrieval(): ...
