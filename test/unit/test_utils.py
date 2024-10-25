import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs
from pandas import testing as pdt

from meteor import utils
from meteor.rsmap import Map
from meteor.testing import MapColumns

NP_RNG = np.random.default_rng()


def omit_nones_in_list(input_list: list) -> list:
    return [x for x in input_list if x]


def test_filter_common_indices() -> None:
    df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2])
    df2 = pd.DataFrame({"B": [4, 5, 6]}, index=[1, 2, 3])
    filtered_df1, filtered_df2 = utils.filter_common_indices(df1, df2)
    assert len(filtered_df1) == 2
    assert len(filtered_df2) == 2


def test_filter_common_indices_empty_intersection() -> None:
    df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2])
    df2 = pd.DataFrame({"B": [4, 5, 6]}, index=[4, 5, 6])
    with pytest.raises(IndexError):
        _, _ = utils.filter_common_indices(df1, df2)


@pytest.mark.parametrize(
    ("dmax_limit", "dmin_limit"),
    [
        (None, None),
        (None, 2.0),
        (8.0, None),
        (8.0, 2.0),
    ],
)
def test_cut_resolution(random_difference_map: Map, dmax_limit: float, dmin_limit: float) -> None:
    dmax_before_cut, dmin_before_cut = random_difference_map.resolution_limits
    expected_dmax_upper_bound: float = max(omit_nones_in_list([dmax_before_cut, dmax_limit]))
    expected_dmin_lower_bound: float = min(omit_nones_in_list([dmin_before_cut, dmin_limit]))

    random_intensities = utils.cut_resolution(
        random_difference_map,
        dmax_limit=dmax_limit,
        dmin_limit=dmin_limit,
    )
    assert len(random_intensities) > 0

    dmax, dmin = random_intensities.resolution_limits
    assert dmax <= expected_dmax_upper_bound
    assert dmin >= expected_dmin_lower_bound


@pytest.mark.parametrize("inplace", [False, True])
def test_canonicalize_amplitudes(
    inplace: bool, random_difference_map: Map, test_map_columns: MapColumns
) -> None:
    # ensure at least one amplitude is negative, one phase is outside [-180,180)
    index_single_hkl = 0
    random_difference_map.loc[index_single_hkl, test_map_columns.amplitude] = -1.0
    random_difference_map.loc[index_single_hkl, test_map_columns.phase] = -470.0

    if inplace:
        canonicalized = random_difference_map
        utils.canonicalize_amplitudes(
            canonicalized,
            amplitude_column=test_map_columns.amplitude,
            phase_column=test_map_columns.phase,
            inplace=inplace,
        )
    else:
        canonicalized = utils.canonicalize_amplitudes(
            random_difference_map,
            amplitude_column=test_map_columns.amplitude,
            phase_column=test_map_columns.phase,
            inplace=inplace,
        )

    assert (canonicalized[test_map_columns.amplitude] >= 0.0).all(), "not all amps positive"
    assert (canonicalized[test_map_columns.phase] >= -180.0).all(), "not all phases > -180"
    assert (canonicalized[test_map_columns.phase] <= 180.0).all(), "not all phases < +180"

    np.testing.assert_almost_equal(
        np.array(np.abs(random_difference_map[test_map_columns.amplitude])),
        np.array(canonicalized[test_map_columns.amplitude]),
    )


def test_average_phase_diff_in_degrees() -> None:
    arr1 = np.array([0.0, 1.0, 1.0, 1.0])
    arr2 = np.array([0.0, 1.0, 1.0, 1.0]) + 1j * np.array([0.0, 0.0, 1.0, 1.0])
    excepted_average_phase_difference = (45.0 * 2) / 4.0
    computed_average_phase_difference = utils.average_phase_diff_in_degrees(arr1, arr2)
    assert np.allclose(computed_average_phase_difference, excepted_average_phase_difference)


def test_average_phase_diff_in_degrees_shape_mismatch() -> None:
    arr1 = np.ones(2)
    arr2 = np.ones(3)
    with pytest.raises(utils.ShapeMismatchError):
        utils.average_phase_diff_in_degrees(arr1, arr2)


def test_complex_array_to_rs_dataseries() -> None:
    carray = np.array([1.0, 0.0, -1.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0, -1.0])
    index = pd.Index(np.arange(4))

    expected_amp = rs.DataSeries(np.ones(4), index=index, name="F").astype(
        rs.StructureFactorAmplitudeDtype(),
    )
    expected_phase = rs.DataSeries([0.0, 90.0, 180.0, -90.0], index=index, name="PHI").astype(
        rs.PhaseDtype(),
    )

    amp, phase = utils.complex_array_to_rs_dataseries(carray, index=index)
    pdt.assert_series_equal(amp, expected_amp)
    pdt.assert_series_equal(phase, expected_phase)


def test_complex_array_to_rs_dataseries_index_mismatch() -> None:
    carray = np.array([1.0]) + 1j * np.array([1.0])
    index = pd.Index(np.arange(2))
    with pytest.raises(utils.ShapeMismatchError):
        utils.complex_array_to_rs_dataseries(carray, index=index)
