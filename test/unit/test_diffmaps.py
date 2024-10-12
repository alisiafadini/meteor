import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    compute_kweights,
)


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
        native_amplitudes_column="NativeAmplitudes",
        derivative_amplitudes_column="DerivativeAmplitudes",
        native_phases_column="NativePhases",
        derivative_phases_column="DerivativePhases",
        sigf_native_column="SigFNative",
        sigf_deriv_column="SigFDeriv",
        use_fixed_kweight=0.5,
    )
    assert isinstance(result, rs.DataSet)
    assert "DFKWeighted" in result.columns


def test_compute_difference_map_vs_analytical(dummy_dataset):

    # Expected complex numbers for native and derivative
    native_complex = np.array(
        [1.0 + 0j, -2.0 + 0j]
    )  # (amplitude 1, phase 0), (amplitude 2, phase 180)
    derivative_complex = np.array(
        [-2.0 + 0j, 3.0 + 0j]
    )  # (amplitude 2, phase 180), (amplitude 3, phase 0)

    # Compute expected complex differences
    delta_complex = derivative_complex - native_complex

    # Expected amplitude and phase differences
    expected_amplitudes = np.abs(delta_complex)
    expected_phases = np.rad2deg(np.angle(delta_complex)) % 360

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
        native_amplitudes_column="NativeAmplitudes",
        derivative_amplitudes_column="DerivativeAmplitudes",
        native_phases_column="NativePhases",
        derivative_phases_column="DerivativePhases",
        sigf_native_column="SigFNative",
        sigf_deriv_column="SigFDeriv",
        use_fixed_kweight=0.5,
    )

    # Correct expected weighted amplitudes (calculated by hand)
    expected_weighted_amplitudes = np.array([1.3247, 1.8280])

    # Compare results, using 4 decimal places for comparison
    np.testing.assert_almost_equal(
        result["DFKWeighted"].values, expected_weighted_amplitudes, decimal=4
    )


def test_kweight_optimization_edge_case():
    # Create a super simple dataset with very small differences between native and derivative amplitudes
    index = pd.MultiIndex.from_arrays(
        [[1, 1], [1, 2], [1, 3]], names=("H", "K", "L")
    ).astype(rs.HKLIndexDtype())
    data = {
        "NativeAmplitudes": np.array([1.0, 1.0]),
        "DerivativeAmplitudes": np.array([1.01, 1.01]),
        "NativePhases": np.array([0.0, 0.0]),  # in degrees
        "DerivativePhases": np.array([0.0, 0.0]),
        "SigFNative": np.array([0.01, 0.01]),
        "SigFDeriv": np.array([0.01, 0.01]),
    }

    # Create a DataSet and add unit cell and space group information
    dataset = rs.DataSet(data, index=index)

    # Add unit cell and space group information
    dataset.cell = gemmi.UnitCell(
        10, 10, 10, 90, 90, 90
    )  # Example unit cell parameters
    dataset.spacegroup = gemmi.SpaceGroup("P 1")  # Simple space group (no symmetry)

    # Run the function with k-weight optimization enabled
    result = compute_kweighted_difference_map(
        dataset=dataset,
        native_amplitudes_column="NativeAmplitudes",
        derivative_amplitudes_column="DerivativeAmplitudes",
        native_phases_column="NativePhases",
        derivative_phases_column="DerivativePhases",
        sigf_native_column="SigFNative",
        sigf_deriv_column="SigFDeriv",
        use_fixed_kweight=False,  # Enable optimization
    )

    # Log the result for checking
    print("Optimized k-weight:", result.attrs.get("kweight"))
    print("DF (unweighted):", result["DF"].values)
    print("DFKWeighted (weighted):", result["DFKWeighted"].values)

    # Since the differences are minimal, we expect the optimization to return a low k-weight value
    assert (
        result.attrs.get("kweight") < 0.1
    )  # Optimized k-weight should be small for this simple case
