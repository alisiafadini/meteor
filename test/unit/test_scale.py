import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor import scale
from meteor.rsmap import Map
from meteor.scale import _compute_anisotropic_scale_factors
from meteor.testing import MapColumns


@pytest.fixture
def miller_dataseries() -> rs.DataSeries:
    miller_indices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    data = np.array([8.0, 4.0, 2.0, 1.0, 1.0], dtype=np.float32)
    return rs.DataSeries(
        data,
        index=pd.MultiIndex.from_tuples(miller_indices, names=["H", "K", "L"]),
    )


def test_compute_anisotropic_scale_factors_smoke(miller_dataseries: rs.DataSeries) -> None:
    # test call signature, valid return
    np_rng = np.random.default_rng()
    random_params: scale.ScaleParameters = tuple(np_rng.normal(size=7))
    scale_factors = _compute_anisotropic_scale_factors(miller_dataseries.index, random_params)
    assert len(scale_factors) == len(miller_dataseries)


def test_compute_scale_factors_identical(miller_dataseries: rs.DataSeries) -> None:
    scale_factors = scale.compute_scale_factors(
        reference_values=miller_dataseries,
        values_to_scale=miller_dataseries,
    )
    assert (scale_factors == 1.0).all()

    equal_uncertainties = miller_dataseries.copy()
    equal_uncertainties[:] = np.ones(len(equal_uncertainties))

    scale_factors = scale.compute_scale_factors(
        reference_values=miller_dataseries,
        values_to_scale=miller_dataseries,
        reference_uncertainties=equal_uncertainties,
        to_scale_uncertainties=equal_uncertainties,
    )
    assert (scale_factors == 1.0).all()


def test_compute_scale_factors_shuffle_indices(miller_dataseries: rs.DataSeries) -> None:
    shuffled_miller_dataseries = miller_dataseries.sample(frac=1)
    scale_factors = scale.compute_scale_factors(
        reference_values=miller_dataseries,
        values_to_scale=shuffled_miller_dataseries,
    )
    assert (scale_factors == 1.0).all()


def test_compute_scale_factors_scalar(miller_dataseries: rs.DataSeries) -> None:
    multiple = 2.0
    doubled_miller_dataseries = miller_dataseries / multiple

    scale_factors = scale.compute_scale_factors(
        reference_values=miller_dataseries,
        values_to_scale=doubled_miller_dataseries,
    )
    np.testing.assert_array_almost_equal(scale_factors, multiple)


def test_compute_scale_factors_anisotropic(miller_dataseries: rs.DataSeries) -> None:
    flat_miller_dataseries = miller_dataseries.copy()
    flat_miller_dataseries[:] = np.ones(len(miller_dataseries))

    scale_factors = scale.compute_scale_factors(
        reference_values=miller_dataseries,
        values_to_scale=flat_miller_dataseries,
    )
    np.testing.assert_array_almost_equal(scale_factors, miller_dataseries.values)


def test_scale_datasets(
    random_difference_map: Map,
    test_map_columns: MapColumns,
) -> None:
    multiple = 2.0
    doubled_difference_map = random_difference_map.copy()
    doubled_difference_map[test_map_columns.amplitude] /= multiple

    scaled = scale.scale_datasets(
        reference_dataset=random_difference_map,
        dataset_to_scale=doubled_difference_map,
        column_to_compare=test_map_columns.amplitude,
        weight_using_uncertainties=False,
    )
    np.testing.assert_array_almost_equal(
        scaled[test_map_columns.amplitude],
        random_difference_map[test_map_columns.amplitude],
    )
    np.testing.assert_array_almost_equal(
        scaled[test_map_columns.phase],
        random_difference_map[test_map_columns.phase],
    )


def test_scale_datasets_with_errors(
    random_difference_map: Map,
    test_map_columns: MapColumns,
) -> None:
    multiple = 2.0
    doubled_difference_map = random_difference_map.copy()
    doubled_difference_map[test_map_columns.amplitude] /= multiple

    scaled = scale.scale_datasets(
        reference_dataset=random_difference_map,
        dataset_to_scale=doubled_difference_map,
        column_to_compare=test_map_columns.amplitude,
        uncertainty_column=str(test_map_columns.uncertainty),
        weight_using_uncertainties=True,
    )
    np.testing.assert_array_almost_equal(
        scaled[test_map_columns.amplitude],
        random_difference_map[test_map_columns.amplitude],
    )
    np.testing.assert_array_almost_equal(
        scaled[test_map_columns.phase],
        random_difference_map[test_map_columns.phase],
    )

    # also make sure we scale the uncertainties
    np.testing.assert_array_almost_equal(
        scaled[test_map_columns.uncertainty] / multiple,
        random_difference_map[test_map_columns.uncertainty],
    )
