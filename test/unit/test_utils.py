from meteor import utils
import reciprocalspaceship as rs
import pytest
import gemmi as gm
import pandas as pd
import numpy as np


def test_resolution_limits(random_intensities: rs.DataSet) -> None:
    dmax, dmin = utils.resolution_limits(random_intensities)
    assert dmax == 10.0
    assert dmin == 1.0


@pytest.mark.parametrize(
    "dmax_limit, dmin_limit",
    [
        (None, None),
        (None, 2.0),
        (8.0, None),
        (8.0, 2.0),
    ],
)
def test_cut_resolution(
    random_intensities: rs.DataSet, dmax_limit: float, dmin_limit: float
) -> None:
    dmax_before_cut, dmin_before_cut = utils.resolution_limits(random_intensities)
    if not dmax_limit:
        expected_max_dmax = dmax_before_cut
    else:
        expected_max_dmax = dmax_limit

    if not dmin_limit:
        expected_min_dmin = dmin_before_cut
    else:
        expected_min_dmin = dmin_limit

    random_intensities = utils.cut_resolution(
        random_intensities, dmax_limit=dmax_limit, dmin_limit=dmin_limit
    )
    assert len(random_intensities) > 0

    dmax, dmin = utils.resolution_limits(random_intensities)
    assert dmax <= expected_max_dmax
    assert dmin >= expected_min_dmin


@pytest.mark.parametrize("inplace", [False, True])
def test_canonicalize_amplitudes(
    inplace: bool, flat_difference_map: rs.DataSet
) -> None:
    amplitude_label = "DF"
    phase_label = "PHIC"

    if inplace:
        canonicalized = flat_difference_map.copy(deep=True)
        utils.canonicalize_amplitudes(
            canonicalized,
            amplitude_label=amplitude_label,
            phase_label=phase_label,
            inplace=inplace,
        )
    else:
        canonicalized = utils.canonicalize_amplitudes(
            flat_difference_map,
            amplitude_label=amplitude_label,
            phase_label=phase_label,
            inplace=inplace,
        )

    assert (canonicalized[amplitude_label] >= 0.0).all()
    assert (canonicalized[phase_label] >= -180.0).all()
    assert (canonicalized[phase_label] <= 180.0).all()

    np.testing.assert_almost_equal(
        np.array(np.abs(flat_difference_map[amplitude_label])),
        np.array(canonicalized[amplitude_label]),
    )


def test_compute_map_from_coefficients(flat_difference_map: rs.DataSet) -> None:
    map = utils.compute_map_from_coefficients(
        map_coefficients=flat_difference_map,
        amplitude_label="DF",
        phase_label="PHIC",
        map_sampling=1,
    )
    assert isinstance(map, gm.Ccp4Map)
    assert map.grid.shape == (6, 6, 6)


@pytest.mark.parametrize("map_sampling", [1, 2, 3, 5])
def test_map_round_trip_ccp4_format(
    map_sampling: int, flat_difference_map: rs.DataSet
) -> None:
    amplitude_label = "DF"
    phase_label = "PHIC"

    utils.canonicalize_amplitudes(
        flat_difference_map,
        amplitude_label=amplitude_label,
        phase_label=phase_label,
        inplace=True,
    )

    map = utils.compute_map_from_coefficients(
        map_coefficients=flat_difference_map,
        amplitude_label=amplitude_label,
        phase_label=phase_label,
        map_sampling=map_sampling,
    )

    _, dmin = utils.resolution_limits(flat_difference_map)

    output_coefficients = utils.compute_coefficients_from_map(
        map=map,
        high_resolution_limit=dmin,
        amplitude_label=amplitude_label,
        phase_label=phase_label,
    )

    pd.testing.assert_frame_equal(
        left=flat_difference_map, right=output_coefficients, atol=1e-3
    )
