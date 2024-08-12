

import reciprocalspaceship as rs
import gemmi as gm
import numpy as np

from pytest import fixture
from meteor import tv


@fixture
def flat_difference_map() -> rs.DataSet:
    """
    A simple 3x3x3 P1 map, random
    """

    params = (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    cell = gm.UnitCell(*params)
    sg_1 = gm.SpaceGroup(1)
    Hall = rs.utils.generate_reciprocal_asu(cell, sg_1, 5.0, anomalous=False)

    h, k, l = Hall.T
    ds = rs.DataSet(
        {
            "H": h,
            "K": k,
            "L": l,
            "DF": np.random.randn(len(h)),
            "PHIC": np.zeros(len(h)),
        },
        spacegroup=sg_1,
        cell=cell,
    ).infer_mtz_dtypes()
    ds.set_index(["H", "K", "L"], inplace=True)

    return ds


def test_tv_denoise_difference_map_smoke(flat_difference_map: rs.DataSet) -> None:

    # test sequence pf specified lambda
    tv.tv_denoise_difference_map(
        difference_map_coefficients=flat_difference_map,
        lambda_values_to_scan=[1.0, 2.0]
    )

    # test golden optimizer
    tv.TV_LAMBDA_RANGE = (1.0, 1.01)
    tv.tv_denoise_difference_map(
        difference_map_coefficients=flat_difference_map,
    )


def test_tv_denoise_difference_map_golden():
    ...

def test_tv_denoise_difference_map_specific_lambdas():
    ...
