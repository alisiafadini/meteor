from pytest import fixture
import reciprocalspaceship as rs
import numpy as np
import gemmi
from meteor.utils import canonicalize_amplitudes


@fixture
def random_intensities() -> rs.DataSet:
    """
    A simple 10x10x10 P1 dataset, with random intensities
    """

    params = (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    cell = gemmi.UnitCell(*params)
    sg_1 = gemmi.SpaceGroup(1)
    Hall = rs.utils.generate_reciprocal_asu(cell, sg_1, 1.0, anomalous=False)

    H, K, L = Hall.T
    ds = rs.DataSet(
        {
            "H": H,
            "K": K,
            "L": L,
            "IMEAN": np.abs(np.random.randn(len(H))),
        },
        spacegroup=sg_1,
        cell=cell,
    ).infer_mtz_dtypes()
    ds.set_index(["H", "K", "L"], inplace=True)

    return ds


@fixture
def flat_difference_map() -> rs.DataSet:
    """
    A simple 3x3x3 P1 map, random
    """

    params = (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    cell = gemmi.UnitCell(*params)
    sg_1 = gemmi.SpaceGroup(1)
    Hall = rs.utils.generate_reciprocal_asu(cell, sg_1, 5.0, anomalous=False)

    H, K, L = Hall.T
    ds = rs.DataSet(
        {
            "H": H,
            "K": K,
            "L": L,
            "DF": np.random.randn(len(H)),
            "PHIC": np.zeros(len(H)),
        },
        spacegroup=sg_1,
        cell=cell,
    ).infer_mtz_dtypes()

    ds.set_index(["H", "K", "L"], inplace=True)
    ds["DF"] = ds["DF"].astype("SFAmplitude")

    canonicalize_amplitudes(
        ds,
        amplitude_label="DF",
        phase_label="PHIC",
        inplace=True,
    )

    return ds
