from meteor import utils
import reciprocalspaceship as rs
import pytest


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
        ]
    )
def test_cut_resolution(random_intensities: rs.DataSet, dmax_limit: float, dmin_limit: float) -> None:

    dmax_before_cut, dmin_before_cut = utils.resolution_limits(random_intensities)
    if not dmax_limit:
        expected_max_dmax = dmax_before_cut
    else:
        expected_max_dmax = dmax_limit

    if not dmin_limit:
        expected_min_dmin = dmin_before_cut
    else:
        expected_min_dmin = dmin_limit

    random_intensities = utils.cut_resolution(random_intensities, dmax_limit=dmax_limit, dmin_limit=dmin_limit)
    assert len(random_intensities) > 0

    dmax, dmin = utils.resolution_limits(random_intensities)
    assert dmax <= expected_max_dmax
    assert dmin >= expected_min_dmin
