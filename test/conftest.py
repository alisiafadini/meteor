from pytest import fixture
import reciprocalspaceship as rs
import numpy as np
import gemmi as gm


@fixture
def random_intensities() -> rs.DataSet:
    """
    A simple 10x10x10 P1 dataset, with random intensities
    """

    params = (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    cell = gm.UnitCell(*params)
    sg_1 = gm.SpaceGroup(1)
    Hall = rs.utils.generate_reciprocal_asu(cell, sg_1, 1.0, anomalous=False)

    h, k, l = Hall.T
    ds = rs.DataSet(
        {
            "H": h,
            "K": k,
            "L": l,
            "IMEAN": np.abs(np.random.randn(len(h))),
        },
        spacegroup=sg_1,
        cell=cell,
    ).infer_mtz_dtypes()
    ds.set_index(["H", "K", "L"], inplace=True)

    return ds