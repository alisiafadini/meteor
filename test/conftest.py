import gemmi
import numpy as np
import reciprocalspaceship as rs
from pytest import fixture

from meteor.utils import canonicalize_amplitudes


@fixture()
def random_difference_map() -> rs.DataSet:
    resolution = 1.0
    cell = gemmi.UnitCell(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    space_group = gemmi.SpaceGroup(1)
    Hall = rs.utils.generate_reciprocal_asu(cell, space_group, resolution, anomalous=False)

    h, k, l = Hall.T
    number_of_reflections = len(h)
    
    ds = rs.DataSet(
        {
            "H": h,
            "K": k,
            "L": l,
            "DF": np.random.randn(number_of_reflections),
            "PHIC": np.random.uniform(-180, 180, size=number_of_reflections),
        },
        spacegroup=space_group,
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