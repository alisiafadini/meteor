from dataclasses import dataclass
from typing import Literal, overload

import gemmi
import numpy as np
import reciprocalspaceship as rs

GEMMI_HIGH_RESOLUTION_BUFFER = 1e-6


@dataclass
class MapLabels:
    amplitude: str
    phase: str


def resolution_limits(dataset: rs.DataSet) -> tuple[float, float]:
    d_hkl = dataset.compute_dHKL()["dHKL"]
    return d_hkl.max(), d_hkl.min()


def cut_resolution(
    dataset: rs.DataSet,
    *,
    dmax_limit: float | None = None,
    dmin_limit: float | None = None,
) -> rs.DataSet:
    d_hkl = dataset.compute_dHKL()["dHKL"]
    if dmax_limit:
        dataset = dataset.loc[(d_hkl <= dmax_limit)]
    if dmin_limit:
        dataset = dataset.loc[(d_hkl >= dmin_limit)]
    return dataset


@overload
def canonicalize_amplitudes(
    dataset: rs.DataSet,
    *,
    amplitude_label: str,
    phase_label: str,
    inplace: Literal[False],
) -> rs.DataSet: ...


@overload
def canonicalize_amplitudes(
    dataset: rs.DataSet,
    *,
    amplitude_label: str,
    phase_label: str,
    inplace: Literal[True],
) -> None: ...


def canonicalize_amplitudes(
    dataset: rs.DataSet,
    *,
    amplitude_label: str,
    phase_label: str,
    inplace: bool = False,
) -> rs.DataSet | None:
    if not inplace:
        dataset = dataset.copy(deep=True)

    negative_amplitude_indices = dataset[amplitude_label] < 0.0
    dataset[amplitude_label] = np.abs(dataset[amplitude_label])
    dataset.loc[negative_amplitude_indices, phase_label] += 180.0

    dataset.canonicalize_phases(inplace=True)

    if not inplace:
        return dataset
    else:
        return None


def numpy_array_to_map(
    array: np.ndarray,
    *,
    spacegroup: str | int | gemmi.SpaceGroup,
    cell: tuple[float, float, float, float, float, float] | gemmi.UnitCell,
) -> gemmi.Ccp4Map:
    ccp4_map = gemmi.Ccp4Map()
    ccp4_map.grid = gemmi.FloatGrid(array.astype(np.float32))

    if isinstance(cell, gemmi.UnitCell):
        ccp4_map.grid.unit_cell = cell
    else:
        ccp4_map.grid.unit_cell.set(*cell)

    if not isinstance(spacegroup, gemmi.SpaceGroup):
        spacegroup = gemmi.SpaceGroup(spacegroup)
    ccp4_map.grid.spacegroup = spacegroup

    return ccp4_map


def compute_map_from_coefficients(
    *,
    map_coefficients: rs.DataSet,
    amplitude_label: str,
    phase_label: str,
    map_sampling: int,
) -> gemmi.Ccp4Map:
    map_coefficients_gemmi_format = map_coefficients.to_gemmi()
    ccp4_map = gemmi.Ccp4Map()
    ccp4_map.grid = map_coefficients_gemmi_format.transform_f_phi_to_map(
        amplitude_label, phase_label, sample_rate=map_sampling
    )
    ccp4_map.update_ccp4_header()

    return ccp4_map


def compute_coefficients_from_map(
    *,
    ccp4_map: gemmi.Ccp4Map,
    high_resolution_limit: float,
    amplitude_label: str,
    phase_label: str,
) -> rs.DataSet:
    # to ensure we include the final shell of reflections, add a small buffer to the resolution

    gemmi_structure_factors = gemmi.transform_map_to_f_phi(ccp4_map.grid, half_l=False)
    data = gemmi_structure_factors.prepare_asu_data(
        dmin=high_resolution_limit - GEMMI_HIGH_RESOLUTION_BUFFER, with_sys_abs=True
    )

    mtz = gemmi.Mtz(with_base=True)
    mtz.spacegroup = gemmi_structure_factors.spacegroup
    mtz.set_cell_for_all(gemmi_structure_factors.unit_cell)
    mtz.add_dataset("FromMap")
    mtz.add_column(amplitude_label, "F")
    mtz.add_column(phase_label, "P")

    mtz.set_data(data)
    mtz.switch_to_asu_hkl()

    return rs.DataSet.from_gemmi(mtz)
