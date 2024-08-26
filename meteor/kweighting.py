from dataclasses import dataclass
from typing import Literal, Sequence, overload

import numpy as np
import reciprocalspaceship as rs
from scipy.optimize import minimize_scalar

from .map_utils import compute_amplitude_fofo_difference
from .settings import KWEIGHTED_MAP_SAMPLING
from .utils import compute_map_from_coefficients
from .validate import negentropy


@dataclass
class KWeightingOptimizationResult:
    optimal_kweight: float
    optimal_negentropy: float

def compute_weights(df: rs.DataSeries, sigdf: rs.DataSeries, kweight: float) -> rs.DataSeries:
    """
    Compute weights for each structure factor based on DeltaF and its uncertainty.
    """
    w = 1 + (sigdf**2 / (sigdf**2).mean()) + kweight * (df**2 / (df**2).mean())
    return w**-1

def compute_kweighted_difference_map(
    data: tuple[rs.DataSeries, rs.DataSeries],
    uncertainties: tuple[rs.DataSeries, rs.DataSeries],
    f_calcs: rs.DataSeries,
    phases: rs.DataSeries,
    kweight: float,
    crystal_params: dict,
) -> rs.DataSet:
    """
    Compute the k-weighted difference map using two datasets, pre-calculated structure factors,
    and the corresponding weights.
    """
    data1, data2 = data
    sigdata1, sigdata2 = uncertainties

    fofo_diff, sig_diffs = compute_amplitude_fofo_difference(data1, data2,
                                                             f_calcs, sigdata1,
                                                             sigdata2)
    weights = compute_weights(fofo_diff, sig_diffs, kweight)
    weighted_diff = fofo_diff * weights

    diff_dataset = rs.DataSet({
        "KWeightedFoFo": weighted_diff,
        "Phases": phases,
    })
    diff_dataset.spacegroup = crystal_params["spacegroup"]
    diff_dataset.cell = crystal_params["cell"]

    return diff_dataset

def _compute_kweighted_map_and_negentropy(
    data: tuple[rs.DataSeries, rs.DataSeries],
    uncertainties: tuple[rs.DataSeries, rs.DataSeries],
    f_calcs: rs.DataSeries,
    phases: rs.DataSeries,
    kweight: float,
    crystal_params: dict,
) -> float:
    """
    Convenience function with hard-coded internal labels
    Compute the k-weighted difference map and return its negentropy.
    """
    weighted_coefficients = compute_kweighted_difference_map(
        data=data,
        uncertainties=uncertainties,
        f_calcs=f_calcs,
        phases=phases,
        kweight=kweight,
        crystal_params=crystal_params,
    )

    weighted_map = compute_map_from_coefficients(
        map_coefficients=weighted_coefficients,
        amplitude_label="KWeightedFoFo",
        phase_label="Phases",
        map_sampling=KWEIGHTED_MAP_SAMPLING
    )

    map_as_array = np.array(weighted_map.grid)
    return -negentropy(map_as_array.flatten())

@overload
def optimize_kweight_for_kweighting(
    data: tuple[rs.DataSeries, rs.DataSeries],
    uncertainties: tuple[rs.DataSeries, rs.DataSeries],
    f_calcs: rs.DataSeries,
    phases: rs.DataSeries,
    crystal_params: dict,
    full_output: Literal[False],
    kweight_values_to_scan: Sequence[float] | None = None,
) -> rs.DataSet:
    ...

@overload
def optimize_kweight_for_kweighting(
    data: tuple[rs.DataSeries, rs.DataSeries],
    uncertainties: tuple[rs.DataSeries, rs.DataSeries],
    f_calcs: rs.DataSeries,
    phases: rs.DataSeries,
    crystal_params: dict,
    full_output: Literal[True],
    kweight_values_to_scan: Sequence[float] | None = None,
) -> tuple[rs.DataSet, KWeightingOptimizationResult]:
    ...

def optimize_kweight_for_kweighting(
    data: tuple[rs.DataSeries, rs.DataSeries],
    uncertainties: tuple[rs.DataSeries, rs.DataSeries],
    f_calcs: rs.DataSeries,
    phases: rs.DataSeries,
    crystal_params: dict,
    full_output: bool = False,
    kweight_values_to_scan: Sequence[float] | np.ndarray | None = None,
) -> rs.DataSet | tuple[rs.DataSet, KWeightingOptimizationResult]:
    """
    Optimize the k-weight parameter for k-weighting to maximize the negentropy of the
    k-weighted difference map.

    This function finds the optimal k-weight parameter that maximizes the negentropy
    of the k-weighted difference map, which is a measure of the map's information
    content. The optimization can be performed either by scanning a set of specified
    k-weight values or by using a golden-section search method to automatically determine
    the optimal value.

    Args:
        data (tuple[rs.DataSeries, rs.DataSeries]):
            A tuple containing the first and second datasets for the difference computation.
        uncertainties (tuple[rs.DataSeries, rs.DataSeries]):
            A tuple containing the uncertainties (standard deviations)
            associated with the two datasets.
        f_calcs (rs.DataSeries):
            Pre-calculated structure factor amplitudes.
        phases (rs.DataSeries):
            Pre-calculated phases.
        crystal_params (dict):
            Dictionary containing the spacegroup and unit cell parameters:
            {"spacegroup": gemmi.SpaceGroup, "cell": gemmi.UnitCell}.
        full_output (bool, optional):
            If `True`, returns both the optimized map and a `KWeightingOptimizationResult`
            object containing the optimal k-weight and the associated negentropy.
            If `False`, only the optimized map is returned. Default is `False`.
        kweight_values_to_scan (Sequence[float] | np.ndarray | None, optional):
            A sequence of k-weight values to explicitly scan for determining the optimal value.
            If `None`, the function uses the golden-section search method to determine the optimal
            k-weight. Default is `None`.

    Returns:
        rs.DataSet | tuple[rs.DataSet, KWeightingOptimizationResult]:
            If `full_output` is `False`, returns an `rs.DataSet` representing the optimized
            k-weighted difference map. If `full_output` is `True`, returns a tuple containing:
            - `rs.DataSet`: The optimized k-weighted difference map.
            - `KWeightingOptimizationResult`: An object containing the optimal k-weight and the
              corresponding negentropy.

    Raises:
        AssertionError:
            If the golden-section search fails to find an optimal k-weight.
        """
    def negentropy_objective(kweight: float) -> float:
        return _compute_kweighted_map_and_negentropy(
            data=data,
            uncertainties=uncertainties,
            f_calcs=f_calcs,
            phases=phases,
            kweight=kweight,
            crystal_params=crystal_params,
        )

    if kweight_values_to_scan is not None:
        optimal_kweight = max(
            kweight_values_to_scan,
            key=lambda k: -negentropy_objective(k)
        )
        highest_negentropy = -negentropy_objective(optimal_kweight)
    else:
        optimizer_result = minimize_scalar(negentropy_objective, method="golden")
        assert optimizer_result.success, "Golden minimization failed to find optimal kweight"
        optimal_kweight = optimizer_result.x
        highest_negentropy = -optimizer_result.fun

    final_coefficients = compute_kweighted_difference_map(
        data=data,
        uncertainties=uncertainties,
        f_calcs=f_calcs,
        phases=phases,
        kweight=optimal_kweight,
        crystal_params=crystal_params,
    )

    if full_output:
        result = KWeightingOptimizationResult(
            optimal_kweight=optimal_kweight, optimal_negentropy=highest_negentropy
        )
        return final_coefficients, result
    else:
        return final_coefficients
