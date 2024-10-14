from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from .rsmap import Map
from .settings import TV_MAP_SAMPLING
from .utils import fitler_common_indices
from .validate import ScalarMaximizer, negentropy

if TYPE_CHECKING:
    import reciprocalspaceship as rs

DEFAULT_KPARAMS_TO_SCAN = np.linspace(0.0, 1.0, 101)


def compute_difference_map(derivative: Map, native: Map) -> Map:
    """
    Computes amplitude and phase differences between native and derivative structure factor sets.

    Parameters
    ----------
    derivative: Map
        the derivative amplitudes, phases, uncertainties
    native: Map
        the native amplitudes, phases, uncertainties

    Returns
    -------
    diffmap: Map
        map corresponding to the complex difference (derivative - native)

    Notes
    -----
    This function computes the complex difference between native and derivative structure factors.
    It converts the amplitude and phase pairs from both the native and derivative structure factor
    sets into complex numbers, computes the difference, and then converts the result back
    into amplitudes and phases.

    If uncertainty columns are provided for both native and derivative data,
    it also propagates the uncertainty of the difference in amplitudes.
    """
    derivative, native = fitler_common_indices(derivative, native)

    delta_complex = derivative.complex_amplitudes - native.complex_amplitudes
    delta = Map.from_structurefactor(delta_complex, index=derivative.index)

    if derivative.has_uncertainties and native.has_uncertainties:
        delta.uncertainties = np.sqrt(derivative.uncertainties**2 + native.uncertainties**2)

    return delta


def compute_kweights(
    difference_map: Map,
    *,
    k_parameter: float,
) -> rs.DataSeries:
    """
    Compute weights for each structure factor based on DeltaF and its uncertainty.

    Parameters
    ----------
    delta: Map
        A map of structure factor differences (DeltaF).
    k_parameter: float
        A scaling factor applied to the squared `df` values in the weight calculation.

    Returns
    -------
    weights: rs.DataSeries
        A series of computed weights, where higher uncertainties and larger differences lead to
        lower weights.
    """
    inverse_weights = (
        1
        + (difference_map.uncertainties**2 / (difference_map.uncertainties**2).mean())
        + k_parameter * (difference_map.amplitudes**2 / (difference_map.amplitudes**2).mean())
    )
    return 1.0 / inverse_weights


def compute_kweighted_difference_map(derivative: Map, native: Map, *, k_parameter: float) -> Map:
    """
    Compute k-weighted derivative - native structure factor map.

    This function first computes the standard difference map using `compute_difference_map`.
    Then, it applies k-weighting to the amplitude differences based on the provided `k_parameter`.

    Assumes amplitudes have already been scaled prior to invoking this function.

    Parameters
    ----------
    derivative: Map
        the derivative amplitudes, phases, uncertainties
    native: Map
        the native amplitudes, phases, uncertainties

    Returns
    -------
    diffmap: Map
        the k-weighted difference map
    """
    diffmap = compute_difference_map(native, derivative)
    weights = compute_kweights(diffmap, k_parameter=k_parameter)
    diffmap.amplitudes *= weights
    return diffmap


def max_negentropy_kweighted_difference_map(
    derivative: Map,
    native: Map,
    *,
    k_parameter_values_to_scan: np.ndarray | Sequence[float] = DEFAULT_KPARAMS_TO_SCAN,
) -> rs.DataSet:
    """
    Compute k-weighted differences between native and derivative amplitudes and phases.

    Determines an "optimal" k_parameter, between 0.0 and 1.0, that maximizes the resulting
    difference map negentropy. Assumes that scaling has already been applied to the amplitudes
    before calling this function.

    Parameters
    ----------
    derivative: Map
        the derivative amplitudes, phases, uncertainties
    native: Map
        the native amplitudes, phases, uncertainties
    k_parameter_values_to_scan : np.ndarray | Sequence[float]
        The values to scan to optimize the k-weighting parameter, by default is 0.00, 0.01 ... 1.00

    Returns
    -------
    kweighted_dataset: rs.DataSet
        dataset with added columns

    opt_k_parameter: float
        optimized k-weighting parameter
    """

    def negentropy_objective(k_parameter: float) -> float:
        kweighted_dataset = compute_kweighted_difference_map(
            native, derivative, k_parameter=k_parameter
        )
        k_weighted_map = kweighted_dataset.to_ccp4_map(map_sampling=TV_MAP_SAMPLING)
        k_weighted_map_array = np.array(k_weighted_map.grid)
        return negentropy(k_weighted_map_array)

    maximizer = ScalarMaximizer(objective=negentropy_objective)
    maximizer.optimize_over_explicit_values(arguments_to_scan=k_parameter_values_to_scan)
    opt_k_parameter = maximizer.argument_optimum

    kweighted_dataset = compute_kweighted_difference_map(
        native, derivative, k_parameter=opt_k_parameter
    )

    return kweighted_dataset, opt_k_parameter
