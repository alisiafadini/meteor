import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    max_negentropy_kweighted_difference_map,
    compute_kweights,
)
from meteor.utils import MapLabels, compute_map_from_coefficients
from meteor.validate import negentropy


# Dummy dataset for testing
@pytest.fixture
def dummy_dataset():
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


def test_compute_difference_map_smoke(dummy_dataset):
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


def test_compute_kweights_smoke(dummy_dataset):
    deltaf = dummy_dataset["NativeAmplitudes"]
    sigdeltaf = dummy_dataset["SigFNative"]
    result = compute_kweights(deltaf, sigdeltaf, kweight=0.5)
    assert isinstance(result, rs.DataSeries)


def test_compute_kweighted_difference_map_smoke(dummy_dataset):
    result = compute_kweighted_difference_map(
        dataset=dummy_dataset,
        kweight=0.5,
        native_amplitudes_column="NativeAmplitudes",
        derivative_amplitudes_column="DerivativeAmplitudes",
        native_phases_column="NativePhases",
        derivative_phases_column="DerivativePhases",
        sigf_native_column="SigFNative",
        sigf_deriv_column="SigFDeriv",
    )
    assert isinstance(result, rs.DataSet)
    assert "DFKWeighted" in result.columns


def test_compute_difference_map_vs_analytical(dummy_dataset):

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

    # Compare the results
    np.testing.assert_almost_equal(result["DF"].values, expected_amplitudes)
    np.testing.assert_almost_equal(result["DPHI"].values, expected_phases)


def test_compute_kweights_vs_analytical():
    # Known deltaF and sigdeltaF values
    deltaf = rs.DataSeries([2.0, 3.0, 4.0])
    sigdeltaf = rs.DataSeries([1.0, 1.0, 1.0])
    kweight = 0.5

    # Expected result (analytically calculated)
    expected_weights = np.array([0.453, 0.406, 0.354])

    # Run the function
    result = compute_kweights(deltaf, sigdeltaf, kweight)

    # Compare results
    np.testing.assert_almost_equal(result.values, expected_weights, decimal=3)


def test_compute_kweighted_difference_map_vs_analytical(dummy_dataset):

    # Run the function with known kweight
    result = compute_kweighted_difference_map(
        dataset=dummy_dataset,
        kweight=0.5,
        native_amplitudes_column="NativeAmplitudes",
        derivative_amplitudes_column="DerivativeAmplitudes",
        native_phases_column="NativePhases",
        derivative_phases_column="DerivativePhases",
        sigf_native_column="SigFNative",
        sigf_deriv_column="SigFDeriv",
    )

    # Correct expected weighted amplitudes (calculated by hand)
    expected_weighted_amplitudes = np.array([1.3247, 1.8280])

    # Compare results, using 4 decimal places for comparison
    np.testing.assert_almost_equal(
        result["DFKWeighted"].values, expected_weighted_amplitudes, decimal=4
    )


def test_kweight_optimization(test_map_labels: MapLabels, noise_free_map: rs.DataSet, very_noisy_map: rs.DataSet):

    very_noisy_map_columns = {
        test_map_labels.amplitude: "F_noisy",
        test_map_labels.phase: "PHIC_noisy",
        test_map_labels.uncertainty: "SIGF_noisy",
    }

    combined_dataset = rs.concat([
        noise_free_map,
        very_noisy_map.rename(columns=very_noisy_map_columns),
    ], axis=1)

    # run the function with k-weight optimization enabled
    result, max_negent_kweight = max_negentropy_kweighted_difference_map(
        dataset=combined_dataset,
        native_amplitudes_column=test_map_labels.amplitude,
        derivative_amplitudes_column=very_noisy_map_columns[test_map_labels.amplitude],
        native_phases_column=test_map_labels.phase,
        derivative_phases_column=very_noisy_map_columns[test_map_labels.phase],
        sigf_native_column=test_map_labels.uncertainty,
        sigf_deriv_column=very_noisy_map_columns[test_map_labels.uncertainty],
    )

    epsilon = 0.01
    kweights_to_scan = [max_negent_kweight - epsilon, max_negent_kweight, max_negent_kweight + epsilon]
    negentropies = []
    
    for kweight in kweights_to_scan:

        kweighted_diffmap = compute_kweighted_difference_map(
            dataset=combined_dataset,
            kweight=kweight,
            native_amplitudes_column=test_map_labels.amplitude,
            derivative_amplitudes_column=very_noisy_map_columns[test_map_labels.amplitude],
            native_phases_column=test_map_labels.phase,
            derivative_phases_column=very_noisy_map_columns[test_map_labels.phase],
            sigf_native_column=test_map_labels.uncertainty,
            sigf_deriv_column=very_noisy_map_columns[test_map_labels.uncertainty],
        )

        realspace_map = compute_map_from_coefficients(
            map_coefficients=kweighted_diffmap,
            amplitude_label="DFKWeighted",
            phase_label="DPHI",
            map_sampling=3,
        )

        map_negentropy = negentropy(np.array(realspace_map.grid))
        negentropies.append(map_negentropy)

    # the optimal k-weight should have the highest negentropy
    print(kweights_to_scan)
    print(negentropies)
    assert negentropies[0] < negentropies[1]
    assert negentropies[2] < negentropies[1]
