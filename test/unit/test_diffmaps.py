import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from numpy.testing import assert_almost_equal

from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    compute_kweights,
    max_negentropy_kweighted_difference_map,
)
from meteor.rsmap import Map
from meteor.utils import MapColumns, compute_map_from_coefficients
from meteor.validate import negentropy


@pytest.fixture
def dummy_derivative() -> Map:
    index = pd.MultiIndex.from_arrays([[1, 1], [1, 2], [1, 3]], names=("H", "K", "L"))
    derivative = {
        "F": np.array([2.0, 3.0]),
        "PHI": np.array([180.0, 0.0]),
        "SIGF": np.array([0.5, 0.5]),
    }
    return Map.from_dict(derivative, index=index)


@pytest.fixture
def dummy_native() -> Map:
    index = pd.MultiIndex.from_arrays([[1, 1], [1, 2], [1, 3]], names=("H", "K", "L"))
    native = {
        "F": np.array([1.0, 2.0]),
        "PHI": np.array([0.0, 180.0]),
        "SIGF": np.array([0.5, 0.5]),
    }
    return Map.from_dict(native, index=index)


def test_compute_difference_map_vs_analytical(dummy_derivative: Map, dummy_native: Map) -> None:
    # Manually calculated expected amplitude and phase differences
    expected_amplitudes = np.array([3.0, 5.0])
    expected_phases = np.array([-180.0, 0.0])
    assert isinstance(dummy_native, Map)
    assert isinstance(dummy_derivative, Map)

    result = compute_difference_map(dummy_derivative, dummy_native)
    assert_almost_equal(result.amplitudes, expected_amplitudes)
    assert_almost_equal(result.phases, expected_phases)


def test_compute_kweights_vs_analytical() -> None:
    deltaf = rs.DataSeries([2.0, 3.0, 4.0])
    phi = rs.DataSeries([0.0, 0.0, 0.0])
    sigdeltaf = rs.DataSeries([1.0, 1.0, 1.0])
    k_parameter = 0.5

    diffmap =  Map.from_dict({"F": deltaf, "PHI": phi, "SIGF": sigdeltaf})
    expected_weights = np.array([0.453, 0.406, 0.354])
    result = compute_kweights(diffmap, k_parameter=k_parameter)
    assert_almost_equal(result.values, expected_weights, decimal=3)


def test_compute_kweighted_difference_map_vs_analytical(dummy_derivative: Map, dummy_native: Map) -> None:
    kwt_diffmap = compute_kweighted_difference_map(dummy_derivative, dummy_native, k_parameter=0.5)
    print("***", kwt_diffmap.uncertainties)

    # expected weighted amplitudes calculated by hand
    expected_weighted_amplitudes = np.array([1.3247, 1.8280])
    assert_almost_equal(kwt_diffmap.amplitudes, expected_weighted_amplitudes, decimal=4)


def test_kweight_optimization(
    noise_free_map: rs.DataSet, noisy_map: rs.DataSet
) -> None:
    _, max_negent_kweight = max_negentropy_kweighted_difference_map(noisy_map, noise_free_map)

    epsilon = 0.01
    k_parameters_to_scan = [
        max(0.0, min(1.0, max_negent_kweight - epsilon)),
        max_negent_kweight,  # Already in range
        max(0.0, min(1.0, max_negent_kweight + epsilon)),
    ]
    negentropies = []
    for k_parameter in k_parameters_to_scan:
        kweighted_diffmap = compute_kweighted_difference_map(
            noisy_map, noise_free_map,
            k_parameter=k_parameter,
        )
        realspace_map = kweighted_diffmap.to_ccp4_map(map_sampling=3)
        map_negentropy = negentropy(np.array(realspace_map.grid))
        negentropies.append(map_negentropy)

    # the optimal k-weight should have the highest negentropy
    assert negentropies[0] <= negentropies[1]
    assert negentropies[2] <= negentropies[1]
