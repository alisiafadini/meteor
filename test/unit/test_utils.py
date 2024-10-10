import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor import utils
from meteor import testing as mt


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
    inplace: bool, random_difference_map: rs.DataSet, diffmap_labels: utils.MapLabels
) -> None:
    # ensure at least one amplitude is negative, one phase is outside [-180,180)
    index_single_hkl = 0
    random_difference_map.loc[index_single_hkl, diffmap_labels.amplitude] = -1.0
    random_difference_map.loc[index_single_hkl, diffmap_labels.phase] = -470.0

    if inplace:
        canonicalized = random_difference_map
        utils.canonicalize_amplitudes(
            canonicalized,
            amplitude_label=diffmap_labels.amplitude,
            phase_label=diffmap_labels.phase,
            inplace=inplace,
        )
    else:
        canonicalized = utils.canonicalize_amplitudes(
            random_difference_map,
            amplitude_label=diffmap_labels.amplitude,
            phase_label=diffmap_labels.phase,
            inplace=inplace,
        )

    assert (canonicalized[diffmap_labels.amplitude] >= 0.0).all(), "not all amplitudes positive"
    assert (canonicalized[diffmap_labels.phase] >= -180.0).all(), "not all phases > -180"
    assert (canonicalized[diffmap_labels.phase] <= 180.0).all(), "not all phases < +180"

    np.testing.assert_almost_equal(
        np.array(np.abs(random_difference_map[diffmap_labels.amplitude])),
        np.array(canonicalized[diffmap_labels.amplitude]),
    )


def test_compute_map_from_coefficients(
    random_difference_map: rs.DataSet, diffmap_labels: utils.MapLabels
) -> None:
    map = utils.compute_map_from_coefficients(
        map_coefficients=random_difference_map,
        amplitude_label=diffmap_labels.amplitude,
        phase_label=diffmap_labels.phase,
        map_sampling=1,
    )
    assert isinstance(map, gemmi.Ccp4Map)


@pytest.mark.parametrize("map_sampling", [1, 2, 2.25, 3, 5])
def test_map_to_coefficients_round_trip(
    map_sampling: int, random_difference_map: rs.DataSet, diffmap_labels: utils.MapLabels
) -> None:
    map = utils.compute_map_from_coefficients(
        map_coefficients=random_difference_map,
        amplitude_label=diffmap_labels.amplitude,
        phase_label=diffmap_labels.phase,
        map_sampling=map_sampling,
    )

    _, dmin = utils.resolution_limits(random_difference_map)

    output_coefficients = utils.compute_coefficients_from_map(
        ccp4_map=map,
        high_resolution_limit=dmin,
        amplitude_label=diffmap_labels.amplitude,
        phase_label=diffmap_labels.phase,
    )

    utils.canonicalize_amplitudes(
        output_coefficients,
        amplitude_label=diffmap_labels.amplitude,
        phase_label=diffmap_labels.phase,
        inplace=True,
    )
    
    pd.testing.assert_series_equal(
        random_difference_map[diffmap_labels.amplitude],
        output_coefficients[diffmap_labels.amplitude],
        atol=1e-3
    )
    mt.assert_phases_allclose(
        random_difference_map[diffmap_labels.phase].to_numpy(),
        output_coefficients[diffmap_labels.phase].to_numpy()
    )
