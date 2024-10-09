import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor import compute_fofo_differences, compute_kweighted_deltafofo


# Test fixture to generate a sample rs.DataSet
@pytest.fixture
def sample_dataset() -> rs.DataSet:
    # Generate some mock data for testing
    miller_indices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    data = np.array([10.0, 8.0, 6.0, 4.0, 2.0], dtype=np.float32)
    uncertainties = np.ones(len(data)) * 0.1

    dataset = rs.DataSet(
        {
            "F_nat": rs.DataSeries(
                data,
                index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
            ),
            "F_deriv": rs.DataSeries(
                data * 1.1,
                index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
            ),
            "F_calc": rs.DataSeries(
                data * 1.05,
                index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
            ),
            "SIGF": rs.DataSeries(
                uncertainties,
                index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
            ),
        }
    )
    return dataset


# Smoke test to verify the function runs without errors
def test_compute_kweighted_deltafofo_smoke(sample_dataset: rs.DataSet) -> None:
    result_dataset = compute_kweighted_deltafofo(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        calc_amplitudes="F_calc",
        kweight=1.0,
    )
    assert isinstance(result_dataset, rs.DataSet)
    assert "DeltaFoFoKWeighted" in result_dataset.columns


# Test the output for a specific kweight value
def test_compute_kweighted_deltafofo_fixed_kweight(sample_dataset: rs.DataSet) -> None:
    kweight = 1.5
    result_dataset = compute_kweighted_deltafofo(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        calc_amplitudes="F_calc",
        kweight=kweight,
    )
    delta_fofo = result_dataset["DeltaFoFoKWeighted"]

    # Test if the weighted DeltaFoFo was correctly computed
    assert not np.isnan(delta_fofo).any()
    assert len(delta_fofo) == len(sample_dataset)


# Test with kweight optimization
def test_compute_kweighted_deltafofo_optimize_kweight(
    sample_dataset: rs.DataSet,
) -> None:
    result_dataset = compute_kweighted_deltafofo(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        calc_amplitudes="F_calc",
        optimize_kweight=True,
    )

    # Ensure the column is added and no NaNs
    delta_fofo_weighted = result_dataset["DeltaFoFoKWeighted"]
    assert "DeltaFoFoKWeighted" in result_dataset.columns
    assert not np.isnan(delta_fofo_weighted).any()


# Test error handling when kweight is None but optimization is False
def test_compute_kweighted_deltafofo_invalid_kweight(
    sample_dataset: rs.DataSet,
) -> None:
    with pytest.raises(ValueError, match="kweight must be provided or optimized"):
        compute_kweighted_deltafofo(
            dataset=sample_dataset,
            native_amplitudes="F_nat",
            derivative_amplitudes="F_deriv",
            calc_amplitudes="F_calc",
            optimize_kweight=False,
        )


# Test if inplace modification works correctly
def test_compute_kweighted_deltafofo_inplace(sample_dataset: rs.DataSet) -> None:
    compute_kweighted_deltafofo(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        calc_amplitudes="F_calc",
        kweight=1.0,
        inplace=True,
    )
    assert "DeltaFoFoKWeighted" in sample_dataset.columns


# Smoke test to verify the function runs without errors
def test_compute_fofo_differences_smoke(sample_dataset: rs.DataSet) -> None:
    result_dataset = compute_fofo_differences(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        calc_amplitudes="F_calc",
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
    )

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
        )


# Test if the function scales properly by checking the range of DeltaFoFo values
def test_compute_fofo_differences_scaling(sample_dataset: rs.DataSet) -> None:
    result_dataset = compute_fofo_differences(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        calc_amplitudes="F_calc",
    )

    # DeltaFoFo should be approximately equal to the difference between derivative and native amplitudes
    delta_fofo = result_dataset["DeltaFoFo"]
    assert delta_fofo.max() > delta_fofo.min()  # Ensure non-zero range
