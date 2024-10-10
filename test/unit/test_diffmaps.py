import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.diffmaps import compute_deltafofo, compute_kweighted_deltafofo


@pytest.fixture
def sample_dataset() -> rs.DataSet:
    # Mock data for testing
    miller_indices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    data = np.array([10.0, 8.0, 6.0, 4.0, 2.0], dtype=np.float32)
    uncertainties_nat = np.ones(len(data)) * 0.1  # Uncertainties for native amplitudes
    uncertainties_deriv = (
        np.ones(len(data)) * 0.2
    )  # Uncertainties for derivative amplitudes
    phases_native = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    phases_derivative = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

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
            "SIGF_nat": rs.DataSeries(
                uncertainties_nat,
                dtype=np.float32,  # Ensure uncertainties are float32
                index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
            ),
            "SIGF_deriv": rs.DataSeries(
                uncertainties_deriv,
                dtype=np.float32,  # Ensure uncertainties are float32
                index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
            ),
            "PHI_nat": rs.DataSeries(
                phases_native,
                dtype=rs.PhaseDtype(),
                index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
            ),
            "PHI_deriv": rs.DataSeries(
                phases_derivative,
                dtype=rs.PhaseDtype(),
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
        native_phases="PHI_nat",
        derivative_phases="PHI_deriv",
        sigf_native="SIGF_nat",
        sigf_deriv="SIGF_deriv",
        kweight=1.0,
    )
    assert isinstance(result_dataset, rs.DataSet)
    assert "DeltaFoFoKWeighted" in result_dataset.columns
    assert "DeltaPhases" in result_dataset.columns


def test_compute_deltafofo_smoke(sample_dataset: rs.DataSet) -> None:
    result_dataset = compute_deltafofo(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        native_phases="PHI_nat",
        derivative_phases="PHI_deriv",
    )
    assert isinstance(result_dataset, rs.DataSet)
    assert "DeltaFoFo" in result_dataset.columns
    assert "DeltaPhases" in result_dataset.columns


# Test the correct output when DeltaFoFo is computed
def test_compute_deltafofo_output(sample_dataset: rs.DataSet) -> None:
    result_dataset = compute_deltafofo(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        native_phases="PHI_nat",
        derivative_phases="PHI_deriv",
    )

    assert result_dataset is not None, "Function returned None when it shouldn't have."

    # Test if the DeltaFoFo (amplitude difference) was correctly computed
    delta_fofo = result_dataset["DeltaFoFo"].to_numpy()

    # Compute expected DeltaFoFo using the updated approach (complex arithmetic)
    native_complex = sample_dataset["F_nat"].to_numpy() * np.exp(
        1j * np.deg2rad(sample_dataset["PHI_nat"].to_numpy())
    )
    derivative_complex = sample_dataset["F_deriv"].to_numpy() * np.exp(
        1j * np.deg2rad(sample_dataset["PHI_deriv"].to_numpy())
    )
    expected_delta_complex = derivative_complex - native_complex
    expected_delta_fofo = np.abs(expected_delta_complex)  # Amplitude differences

    np.testing.assert_array_almost_equal(delta_fofo, expected_delta_fofo, decimal=5)

    # Test if the DeltaPhases were correctly computed
    delta_phases = result_dataset["DeltaPhases"].to_numpy(dtype=np.float32)

    # The expected phase difference must be based on the complex difference
    expected_delta_phases = np.angle(
        expected_delta_complex, deg=True
    )  # Convert back to degrees

    np.testing.assert_array_almost_equal(delta_phases, expected_delta_phases, decimal=5)


# Test in-place modification
def test_compute_deltafofo_inplace(sample_dataset: rs.DataSet) -> None:
    compute_deltafofo(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        native_phases="PHI_nat",
        derivative_phases="PHI_deriv",
        inplace=True,
    )

    # Ensure the DeltaFoFo and DeltaPhases columns exist in the modified dataset
    assert "DeltaFoFo" in sample_dataset.columns
    assert "DeltaPhases" in sample_dataset.columns

    # Compute expected DeltaFoFo and DeltaPhases using the new complex approach
    native_complex = sample_dataset["F_nat"].to_numpy() * np.exp(
        1j * np.deg2rad(sample_dataset["PHI_nat"].to_numpy())
    )
    derivative_complex = sample_dataset["F_deriv"].to_numpy() * np.exp(
        1j * np.deg2rad(sample_dataset["PHI_deriv"].to_numpy())
    )
    expected_delta_complex = derivative_complex - native_complex

    expected_delta_fofo = np.abs(expected_delta_complex)
    expected_delta_phases = np.angle(expected_delta_complex, deg=True)

    # Assert DeltaFoFo values
    delta_fofo = sample_dataset["DeltaFoFo"].to_numpy()
    np.testing.assert_array_almost_equal(delta_fofo, expected_delta_fofo)

    # Assert DeltaPhases values
    delta_phases = sample_dataset["DeltaPhases"].to_numpy(dtype=np.float32)
    np.testing.assert_array_almost_equal(delta_phases, expected_delta_phases, decimal=5)


# Test that no dataset is returned when inplace=True
def test_compute_deltafofo_inplace_return(sample_dataset: rs.DataSet) -> None:
    result = compute_deltafofo(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",  # Fixed typo here
        native_phases="PHI_nat",
        derivative_phases="PHI_deriv",
        inplace=True,
    )
    assert result is None  # Should return None when inplace=True


# Test handling missing columns (e.g. missing native amplitude column)
def test_compute_deltafofo_missing_column(sample_dataset: rs.DataSet) -> None:
    with pytest.raises(KeyError):
        compute_deltafofo(
            dataset=sample_dataset.drop(columns=["F_nat"]),
            native_amplitudes="F_nat",
            derivative_amplitudes="F_deriv",
            native_phases="PHI_nat",
            derivative_phases="PHI_deriv",
        )


def test_compute_deltafofo_range(sample_dataset: rs.DataSet) -> None:
    result_dataset = compute_deltafofo(
        dataset=sample_dataset,
        native_amplitudes="F_nat",
        derivative_amplitudes="F_deriv",
        native_phases="PHI_nat",
        derivative_phases="PHI_deriv",
    )

    # Ensure the function returns a dataset
    assert result_dataset is not None, "Function returned None when it shouldn't have."

    delta_fofo = result_dataset["DeltaFoFo"].to_numpy()

    # Ensure that the DeltaFoFo values are not all the same (i.e. there's a range)
    assert (
        delta_fofo.max() > delta_fofo.min()
    ), "DeltaFoFo values should have a non-zero range."

    # Compute expected DeltaFoFo using complex numbers
    native_complex = sample_dataset["F_nat"].to_numpy() * np.exp(
        1j * np.deg2rad(sample_dataset["PHI_nat"].to_numpy())
    )
    derivative_complex = sample_dataset["F_deriv"].to_numpy() * np.exp(
        1j * np.deg2rad(sample_dataset["PHI_deriv"].to_numpy())
    )
    expected_delta_complex = derivative_complex - native_complex
    expected_delta_fofo = np.abs(expected_delta_complex)

    # Assert that the DeltaFoFo values match expected values within tolerance
    np.testing.assert_array_almost_equal(delta_fofo, expected_delta_fofo, decimal=5)
