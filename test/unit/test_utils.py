import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs
from pandas import testing as pdt

from meteor import testing as mt
from meteor import utils


def omit_nones_in_list(input_list: list) -> list:
    return [x for x in input_list if x]


def test_resolution_limits(random_difference_map: rs.DataSet) -> None:
    dmax, dmin = utils.resolution_limits(random_difference_map)
    assert dmax == 10.0
    assert dmin == 1.0


@pytest.mark.parametrize(
    ("dmax_limit", "dmin_limit"),
    [
        (None, None),
        (None, 2.0),
        (8.0, None),
        (8.0, 2.0),
    ],
)
def test_cut_resolution(
    random_difference_map: rs.DataSet, dmax_limit: float, dmin_limit: float
) -> None:
    dmax_before_cut, dmin_before_cut = utils.resolution_limits(random_difference_map)
    expected_dmax_upper_bound: float = max(omit_nones_in_list([dmax_before_cut, dmax_limit]))
    expected_dmin_lower_bound: float = min(omit_nones_in_list([dmin_before_cut, dmin_limit]))

    random_intensities = utils.cut_resolution(
        random_difference_map, dmax_limit=dmax_limit, dmin_limit=dmin_limit
    )
    assert len(random_intensities) > 0

    dmax, dmin = utils.resolution_limits(random_intensities)
    assert dmax <= expected_dmax_upper_bound
    assert dmin >= expected_dmin_lower_bound


@pytest.mark.parametrize("inplace", [False, True])
def test_canonicalize_amplitudes(
    inplace: bool, random_difference_map: rs.DataSet, test_diffmap_columns: utils.MapColumns
) -> None:
    # ensure at least one amplitude is negative, one phase is outside [-180,180)
    index_single_hkl = 0
    random_difference_map.loc[index_single_hkl, test_diffmap_columns.amplitude] = -1.0
    random_difference_map.loc[index_single_hkl, test_diffmap_columns.phase] = -470.0

    if inplace:
        canonicalized = random_difference_map
        utils.canonicalize_amplitudes(
            canonicalized,
            amplitude_label=test_diffmap_columns.amplitude,
            phase_label=test_diffmap_columns.phase,
            inplace=inplace,
        )
    else:
        canonicalized = utils.canonicalize_amplitudes(
            random_difference_map,
            amplitude_label=test_diffmap_columns.amplitude,
            phase_label=test_diffmap_columns.phase,
            inplace=inplace,
        )

    assert (
        canonicalized[test_diffmap_columns.amplitude] >= 0.0
    ).all(), "not all amplitudes positive"
    assert (canonicalized[test_diffmap_columns.phase] >= -180.0).all(), "not all phases > -180"
    assert (canonicalized[test_diffmap_columns.phase] <= 180.0).all(), "not all phases < +180"

    np.testing.assert_almost_equal(
        np.array(np.abs(random_difference_map[test_diffmap_columns.amplitude])),
        np.array(canonicalized[test_diffmap_columns.amplitude]),
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


def test_rs_dataseries_to_complex_array() -> None:
    index = pd.Index(np.arange(4))
    amp = rs.DataSeries(np.ones(4), index=index)
    phase = rs.DataSeries(np.arange(4) * 90.0, index=index)

    carray = utils.rs_dataseries_to_complex_array(amp, phase)
    expected = np.array([1.0, 0.0, -1.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0, -1.0])

    np.testing.assert_almost_equal(carray, expected)


def test_rs_dataseries_to_complex_array_index_mismatch() -> None:
    amp = rs.DataSeries(np.ones(4), index=[0, 1, 2, 3])
    phase = rs.DataSeries(np.arange(4) * 90.0, index=[1, 2, 3, 4])
    with pytest.raises(utils.ShapeMismatchError):
        utils.rs_dataseries_to_complex_array(amp, phase)


def test_complex_array_to_rs_dataseries() -> None:
    carray = np.array([1.0, 0.0, -1.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0, -1.0])
    index = pd.Index(np.arange(4))

    expected_amp = rs.DataSeries(np.ones(4), index=index).astype(rs.StructureFactorAmplitudeDtype())
    expected_phase = rs.DataSeries([0.0, 90.0, 180.0, -90.0], index=index).astype(rs.PhaseDtype())

    amp, phase = utils.complex_array_to_rs_dataseries(carray, index)
    pdt.assert_series_equal(amp, expected_amp)
    pdt.assert_series_equal(phase, expected_phase)


def test_complex_array_to_rs_dataseries_index_mismatch() -> None:
    carray = np.array([1.0]) + 1j * np.array([1.0])
    index = pd.Index(np.arange(2))
    with pytest.raises(utils.ShapeMismatchError):
        utils.complex_array_to_rs_dataseries(carray, index)


def test_complex_array_dataseries_roundtrip() -> None:
    n = 5
    carray = np.random.randn(n) + 1j * np.random.randn(n)
    indices = pd.Index(np.arange(n))

    ds_amplitudes, ds_phases = utils.complex_array_to_rs_dataseries(carray, indices)

    assert isinstance(ds_amplitudes, rs.DataSeries)
    assert isinstance(ds_phases, rs.DataSeries)

    assert ds_amplitudes.dtype == rs.StructureFactorAmplitudeDtype()
    assert ds_phases.dtype == rs.PhaseDtype()

    pdt.assert_index_equal(ds_amplitudes.index, indices)
    pdt.assert_index_equal(ds_phases.index, indices)

    carray2 = utils.rs_dataseries_to_complex_array(ds_amplitudes, ds_phases)
    np.testing.assert_almost_equal(carray, carray2, decimal=5)


def test_compute_map_from_coefficients(
    random_difference_map: rs.DataSet, test_diffmap_columns: utils.MapColumns
) -> None:
    diffmap = utils.compute_map_from_coefficients(
        map_coefficients=random_difference_map,
        amplitude_label=test_diffmap_columns.amplitude,
        phase_label=test_diffmap_columns.phase,
        map_sampling=1,
    )
    assert isinstance(diffmap, gemmi.Ccp4Map)


@pytest.mark.parametrize("map_sampling", [1, 2, 2.25, 3, 5])
def test_map_to_coefficients_round_trip(
    map_sampling: int, random_difference_map: rs.DataSet, test_diffmap_columns: utils.MapColumns
) -> None:
    realspace_map = utils.compute_map_from_coefficients(
        map_coefficients=random_difference_map,
        amplitude_label=test_diffmap_columns.amplitude,
        phase_label=test_diffmap_columns.phase,
        map_sampling=map_sampling,
    )

    _, dmin = utils.resolution_limits(random_difference_map)

    output_coefficients = utils.compute_coefficients_from_map(
        ccp4_map=realspace_map,
        high_resolution_limit=dmin,
        amplitude_label=test_diffmap_columns.amplitude,
        phase_label=test_diffmap_columns.phase,
    )

    utils.canonicalize_amplitudes(
        output_coefficients,
        amplitude_label=test_diffmap_columns.amplitude,
        phase_label=test_diffmap_columns.phase,
        inplace=True,
    )

    pd.testing.assert_series_equal(
        random_difference_map[test_diffmap_columns.amplitude],
        output_coefficients[test_diffmap_columns.amplitude],
        atol=1e-3,
    )
    mt.assert_phases_allclose(
        random_difference_map[test_diffmap_columns.phase].to_numpy(),
        output_coefficients[test_diffmap_columns.phase].to_numpy(),
    )
