import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.diffmaps import compute_fofo_differences, compute_kweighted_deltafofo


@pytest.fixture
def sample_dataset() -> rs.DataSet:
    # Mock data for testing
    miller_indices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    data = np.array([10.0, 8.0, 6.0, 4.0, 2.0], dtype=np.float32)
    uncertainties_nat = np.ones(len(data)) * 0.1  # Uncertainties for native amplitudes
    uncertainties_deriv = (
        np.ones(len(data)) * 0.2
    )  # Uncertainties for derivative amplitudes

    dataset = rs.DataSet(
        {
            "F_nat": rs.DataSeries(
                data,
                dtype=rs.StructureFactorAmplitudeDtype(),
                index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
            ),
            "F_deriv": rs.DataSeries(
                data * 1.1,
                dtype=rs.StructureFactorAmplitudeDtype(),
                index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
            ),
            "F_calc": rs.DataSeries(
                data * 1.05,
                dtype=rs.StructureFactorAmplitudeDtype(),
                index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
            ),
            "SIGF_nat": rs.DataSeries(
                uncertainties_nat,
                index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
            ),
            "SIGF_deriv": rs.DataSeries(
                uncertainties_deriv,
                index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
            ),
        }
    )
    return dataset


# Smoke tests
def test_compute_kweighted_deltafofo_smoke(sample_dataset: rs.DataSet) -> None:
    result_dataset = compute_kweighted_deltafofo(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        calc_amplitudes="F_calc",
        sigf_native="SIGF_nat",
        sigf_deriv="SIGF_deriv",
        kweight=1.0,
    )
    assert isinstance(result_dataset, rs.DataSet)
    assert "DeltaFoFoKWeighted" in result_dataset.columns


def test_compute_fofo_differences_smoke(sample_dataset: rs.DataSet) -> None:
    result_dataset = compute_fofo_differences(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        calc_amplitudes="F_calc",
        sigf_native="SIGF_nat",
        sigf_deriv="SIGF_deriv",
    )
    assert isinstance(result_dataset, rs.DataSet)
    assert "DeltaFoFo" in result_dataset.columns


# Test the correct output when DeltaFoFo is computed
def test_compute_fofo_differences_output(sample_dataset: rs.DataSet) -> None:
    result_dataset = compute_fofo_differences(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        calc_amplitudes="F_calc",
        sigf_native="SIGF_nat",
        sigf_deriv="SIGF_deriv",
    )

    assert result_dataset is not None, "Function returned None when it shouldn't have."

    # Test if the DeltaFoFo was correctly computed
    delta_fofo = result_dataset["DeltaFoFo"]
    expected_delta_fofo = sample_dataset["F_deriv"] - sample_dataset["F_nat"]

    np.testing.assert_array_almost_equal(delta_fofo, expected_delta_fofo)


# Test in-place modification of the dataset
def test_compute_fofo_differences_inplace(sample_dataset: rs.DataSet) -> None:
    compute_fofo_differences(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        calc_amplitudes="F_calc",
        sigf_native="SIGF_nat",
        sigf_deriv="SIGF_deriv",
        inplace=True,
    )

    # Ensure the DeltaFoFo column exists in the modified dataset
    assert "DeltaFoFo" in sample_dataset.columns

    # Check that the DeltaFoFo values are correct
    expected_delta_fofo = sample_dataset["F_deriv"] - sample_dataset["F_nat"]
    np.testing.assert_array_almost_equal(
        sample_dataset["DeltaFoFo"], expected_delta_fofo
    )


# Test that no dataset is returned when inplace=True
def test_compute_fofo_differences_inplace_return(sample_dataset: rs.DataSet) -> None:
    result = compute_fofo_differences(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        calc_amplitudes="F_calc",
        sigf_native="SIGF_nat",
        sigf_deriv="SIGF_deriv",
        inplace=True,
    )
    assert result is None  # Should return None when inplace=True


# Test handling missing columns (e.g., missing native amplitude column)
def test_compute_fofo_differences_missing_column(sample_dataset: rs.DataSet) -> None:
    with pytest.raises(KeyError):
        compute_fofo_differences(
            dataset=sample_dataset.drop(columns=["F_nat"]),
            native_amplitudes="F_nat",
            derivative_amplitudes="F_deriv",
            calc_amplitudes="F_calc",
            sigf_native="SIGF_nat",
            sigf_deriv="SIGF_deriv",
        )


# Test if the function scales properly by checking the range of DeltaFoFo values
def test_compute_fofo_differences_scaling(sample_dataset: rs.DataSet) -> None:
    result_dataset = compute_fofo_differences(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        calc_amplitudes="F_calc",
        sigf_native="SIGF_nat",
        sigf_deriv="SIGF_deriv",
    )

    # DeltaFoFo should be approximately equal to the
    # difference between derivative and native amplitudes
    assert result_dataset is not None, "Function returned None when it shouldn't have."

    delta_fofo = result_dataset["DeltaFoFo"]
    assert delta_fofo.max() > delta_fofo.min()  # Ensure non-zero range
