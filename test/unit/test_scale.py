import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor import scale


def generate_mock_dataset(miller_indices, data):
    return rs.DataSeries(
        data, index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"])
    )


@pytest.fixture
def identical_datasets():
    miller_indices = [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1)]
    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    return generate_mock_dataset(miller_indices, data), generate_mock_dataset(
        miller_indices, data
    )


@pytest.fixture
def different_datasets():
    miller_indices = [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1)]
    reference_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    scale_data = np.array([15.0, 25.0, 35.0, 45.0], dtype=np.float32)
    return generate_mock_dataset(miller_indices, reference_data), generate_mock_dataset(
        miller_indices, scale_data
    )


def test_miller_indices_mismatch():
    miller_indices_1 = [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1)]
    miller_indices_2 = [(0, 0, 1), (1, 0, 0), (0, 1, 1), (1, 1, 1)]
    data_1 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    data_2 = np.array([15.0, 25.0, 35.0, 45.0], dtype=np.float32)

    reference = generate_mock_dataset(miller_indices_1, data_1)
    dataset_to_scale = generate_mock_dataset(miller_indices_2, data_2)

    with pytest.raises(
        AssertionError,
        match="Miller indices of reference and dataset_to_scale do not match.",
    ):
        scale.scale_structure_factors(reference, dataset_to_scale)


@pytest.mark.parametrize("inplace", [True, False])
def test_scale_structure_factors_identical(identical_datasets, inplace):
    reference, dataset_to_scale = identical_datasets
    original_data = dataset_to_scale.copy()

    result = scale.scale_structure_factors(reference, dataset_to_scale, inplace=inplace)

    if inplace:
        np.testing.assert_array_almost_equal(
            dataset_to_scale.to_numpy(), original_data.to_numpy()
        )
        assert result is None
    else:
        np.testing.assert_array_almost_equal(
            result.to_numpy(), original_data.to_numpy()
        )


@pytest.mark.parametrize("inplace", [True, False])
def test_scale_structure_factors_different(different_datasets, inplace):
    reference, dataset_to_scale = different_datasets
    # should correctly recover from + 5 scale
    if inplace:
        scale.scale_structure_factors(reference, dataset_to_scale, inplace=inplace)
        np.testing.assert_array_almost_equal(
            dataset_to_scale.to_numpy(), reference.to_numpy()
        )
    else:
        result = scale.scale_structure_factors(
            reference, dataset_to_scale, inplace=inplace
        )
        assert result is not None
        np.testing.assert_array_almost_equal(result.to_numpy(), reference.to_numpy())
