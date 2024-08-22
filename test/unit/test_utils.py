import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor import utils


def test_resolution_limits(random_difference_map: rs.DataSet) -> None:
    dmax, dmin = utils.resolution_limits(random_difference_map)
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
    random_difference_map: rs.DataSet, dmax_limit: float, dmin_limit: float
) -> None:
    dmax_before_cut, dmin_before_cut = utils.resolution_limits(random_difference_map)
    if not dmax_limit:
        expected_max_dmax = dmax_before_cut
    else:
        expected_max_dmax = dmax_limit

    if not dmin_limit:
        expected_min_dmin = dmin_before_cut
    else:
        expected_min_dmin = dmin_limit

    random_intensities = utils.cut_resolution(
        random_difference_map, dmax_limit=dmax_limit, dmin_limit=dmin_limit
    )
    assert len(random_intensities) > 0

    dmax, dmin = utils.resolution_limits(random_intensities)
    assert dmax <= expected_max_dmax
    assert dmin >= expected_min_dmin


@pytest.mark.parametrize("inplace", [False, True])
def test_canonicalize_amplitudes(inplace: bool, random_difference_map: rs.DataSet) -> None:
    amplitude_label = "DF"
    phase_label = "PHIC"

    if inplace:
        canonicalized = random_difference_map
        utils.canonicalize_amplitudes(
            canonicalized,
            amplitude_label=amplitude_label,
            phase_label=phase_label,
            inplace=inplace,
        )
    else:
        canonicalized = utils.canonicalize_amplitudes(
            random_difference_map,
            amplitude_label=amplitude_label,
            phase_label=phase_label,
            inplace=inplace,
        )

    assert (canonicalized[amplitude_label] >= 0.0).all(), "not all amplitudes positive"
    assert (canonicalized[phase_label] >= -180.0).all(), "not all phases > -180"
    assert (canonicalized[phase_label] <= 180.0).all(), "not all phases < +180"

    np.testing.assert_almost_equal(
        np.array(np.abs(random_difference_map[amplitude_label])),
        np.array(canonicalized[amplitude_label]),
    )


def test_compute_map_from_coefficients(random_difference_map: rs.DataSet) -> None:
    map = utils.compute_map_from_coefficients(
        map_coefficients=random_difference_map,
        amplitude_label="DF",
        phase_label="PHIC",
        map_sampling=1,
    )
    assert isinstance(map, gemmi.Ccp4Map)


@pytest.mark.parametrize("map_sampling", [1, 2, 2.25, 3, 5])
def test_map_to_coefficients_round_trip(map_sampling: int, random_difference_map: rs.DataSet) -> None:

    # TODO fix this
    amplitude_label = "DF"
    phase_label = "PHIC"

    map = utils.compute_map_from_coefficients(
        map_coefficients=random_difference_map,
        amplitude_label=amplitude_label,
        phase_label=phase_label,
        map_sampling=map_sampling,
    )

    _, dmin = utils.resolution_limits(random_difference_map)

    output_coefficients = utils.compute_coefficients_from_map(
        ccp4_map=map,
        high_resolution_limit=dmin,
        amplitude_label=amplitude_label,
        phase_label=phase_label,
    )

    utils.canonicalize_amplitudes(
        output_coefficients,
        amplitude_label=amplitude_label,
        phase_label=phase_label,
        inplace=True
    )
    pd.testing.assert_frame_equal(left=random_difference_map, right=output_coefficients, atol=0.5)
