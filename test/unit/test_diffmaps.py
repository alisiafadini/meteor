import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    compute_kweights,
    max_negentropy_kweighted_difference_map,
)
from meteor.utils import MapColumns, compute_map_from_coefficients
from meteor.validate import negentropy


@pytest.fixture
def dummy_dataset() -> rs.DataSet:
    index = pd.MultiIndex.from_arrays([[1, 1], [1, 2], [1, 3]], names=("H", "K", "L"))
    data = {
        "NativeAmplitudes": np.array([1.0, 2.0]),
        "DerivativeAmplitudes": np.array([2.0, 3.0]),
        "NativePhases": np.array([0.0, 180.0]),  # in degrees
        "DerivativePhases": np.array([180.0, 0.0]),
        "SigFNative": np.array([0.5, 0.5]),
        "SigFDeriv": np.array([0.5, 0.5]),
    }
    return rs.DataSet(data, index=index)


def test_compute_difference_map_smoke(dummy_dataset: rs.DataSet) -> None:
    result = compute_difference_map(
        dataset=dummy_dataset,
        native_amplitudes_column="NativeAmplitudes",
        derivative_amplitudes_column="DerivativeAmplitudes",
        native_phases_column="NativePhases",
        derivative_phases_column="DerivativePhases",
    )
    assert isinstance(result, rs.DataSet)
    assert "DF" in result.columns
    assert "DPHI" in result.columns


def test_compute_kweights_smoke(dummy_dataset: rs.DataSet) -> None:
    deltaf = dummy_dataset["NativeAmplitudes"]
    sigdeltaf = dummy_dataset["SigFNative"]
    result = compute_kweights(deltaf, sigdeltaf, k_parameter=0.5)
    assert isinstance(result, rs.DataSeries)


def test_compute_kweighted_difference_map_smoke(dummy_dataset: rs.DataSet) -> None:
    result = compute_kweighted_difference_map(
        dataset=dummy_dataset,
        k_parameter=0.5,
        native_amplitudes_column="NativeAmplitudes",
        native_phases_column="NativePhases",
        native_uncertainty_column="SigFNative",
        derivative_amplitudes_column="DerivativeAmplitudes",
        derivative_phases_column="DerivativePhases",
        derivative_uncertainty_column="SigFDeriv",
    )
    assert isinstance(result, rs.DataSet)
    assert "DF_KWeighted" in result.columns


def test_compute_difference_map_vs_analytical(dummy_dataset: rs.DataSet) -> None:
    # Manually calculated expected amplitude and phase differences
    expected_amplitudes = np.array([3.0, 5.0])
    expected_phases = np.array([180.0, 0.0])

    # Run the function
    result = compute_difference_map(
        dataset=dummy_dataset,
        native_amplitudes_column="NativeAmplitudes",
        derivative_amplitudes_column="DerivativeAmplitudes",
        native_phases_column="NativePhases",
        derivative_phases_column="DerivativePhases",
    )

    np.testing.assert_almost_equal(result["DF"].values, expected_amplitudes)
    np.testing.assert_almost_equal(result["DPHI"].values, expected_phases)


def test_compute_kweights_vs_analytical() -> None:
    deltaf = rs.DataSeries([2.0, 3.0, 4.0])
    sigdeltaf = rs.DataSeries([1.0, 1.0, 1.0])
    k_parameter = 0.5

    expected_weights = np.array([0.453, 0.406, 0.354])
    result = compute_kweights(deltaf, sigdeltaf, k_parameter)
    np.testing.assert_almost_equal(result.values, expected_weights, decimal=3)


def test_compute_kweighted_difference_map_vs_analytical(
    dummy_dataset: rs.DataSet,
) -> None:
    result = compute_kweighted_difference_map(
        dataset=dummy_dataset,
        k_parameter=0.5,
        native_amplitudes_column="NativeAmplitudes",
        native_phases_column="NativePhases",
        derivative_amplitudes_column="DerivativeAmplitudes",
        derivative_phases_column="DerivativePhases",
        native_uncertainty_column="SigFNative",
        derivative_uncertainty_column="SigFDeriv",
    )

    # expected weighted amplitudes calculated by hand
    expected_weighted_amplitudes = np.array([1.3247, 1.8280])

    np.testing.assert_almost_equal(
        result["DF_KWeighted"].values, expected_weighted_amplitudes, decimal=4
    )


def test_kweight_optimization(
    test_map_columns: MapColumns, noise_free_map: rs.DataSet, noisy_map: rs.DataSet
) -> None:
    noisy_map_columns = {
        test_map_columns.amplitude: "F_noisy",
        test_map_columns.phase: "PHIC_noisy",
        test_map_columns.uncertainty: "SIGF_noisy",
    }

    combined_dataset = rs.concat(
        [
            noise_free_map,
            noisy_map.rename(columns=noisy_map_columns),
        ],
        axis=1,
    )

    _, max_negent_kweight = max_negentropy_kweighted_difference_map(
        combined_dataset,
        native_amplitudes_column=test_map_columns.amplitude,
        native_phases_column=test_map_columns.phase,
        native_uncertainty_column=test_map_columns.uncertainty,
        derivative_amplitudes_column=noisy_map_columns[test_map_columns.amplitude],
        derivative_phases_column=noisy_map_columns[test_map_columns.phase],
        derivative_uncertainty_column=noisy_map_columns[test_map_columns.uncertainty],
    )

    epsilon = 0.01
    k_parameters_to_scan = [
        max(0.0, min(1.0, max_negent_kweight - epsilon)),
        max_negent_kweight,  # Already in range
        max(0.0, min(1.0, max_negent_kweight + epsilon)),
    ]
    negentropies = []

    for k_parameter in k_parameters_to_scan:
        kweighted_diffmap = compute_kweighted_difference_map(
            dataset=combined_dataset,
            k_parameter=k_parameter,
            native_amplitudes_column=test_map_columns.amplitude,
            native_phases_column=test_map_columns.phase,
            native_uncertainty_column=test_map_columns.uncertainty,
            derivative_amplitudes_column=noisy_map_columns[test_map_columns.amplitude],
            derivative_phases_column=noisy_map_columns[test_map_columns.phase],
            derivative_uncertainty_column=noisy_map_columns[
                test_map_columns.uncertainty
            ],
        )

        realspace_map = compute_map_from_coefficients(
            map_coefficients=kweighted_diffmap,
            amplitude_label="DF_KWeighted",
            phase_label="DPHI_KWeighted",
            map_sampling=3,
        )

        map_negentropy = negentropy(np.array(realspace_map.grid))
        negentropies.append(map_negentropy)

    # the optimal k-weight should have the highest negentropy
    assert negentropies[0] <= negentropies[1]
    assert negentropies[2] <= negentropies[1]
