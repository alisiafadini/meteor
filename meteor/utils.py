from dataclasses import dataclass
from typing import Literal, overload

import gemmi
import numpy as np
import pandas as pd
import reciprocalspaceship as rs
from pandas.testing import assert_index_equal

GEMMI_HIGH_RESOLUTION_BUFFER = 1e-6


class ShapeMismatchError(Exception): ...


@dataclass
class MapLabels:
    amplitude: str
    phase: str
    uncertainty: str | None = None


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


def rs_dataseries_to_complex_array(
    amplitudes: rs.DataSeries, phases: rs.DataSeries
) -> np.ndarray:
    """
    Convert structure factors from polar (amplitude/phase) to Cartisian (x + iy).

    Parameters
    ----------
    amplitudes: DataSeries
        with StructureFactorAmplitudeDtype
    phases: DataSeries
        with PhaseDtype

    Returns
    -------
    complex_structure_factors: np.ndarray
        with dtype complex128

    Raises
    ------
    ValueError
        if the indices of `amplitudes` and `phases` do not match
    """
    try:
        assert_index_equal(amplitudes.index, phases.index)
    except AssertionError as exptn:
        raise ShapeMismatchError(
            f"indices for `amplitudes` and `phases` don't match: {amplitudes.index} {phases.index}",
            " To safely cast to a single complex array, pass DataSeries with a common set of",
            " indices. One possible way: Series.align(other, join='inner', axis=0).",
            exptn,
        )
    complex_structure_factors = amplitudes.to_numpy() * np.exp(
        1j * np.deg2rad(phases.to_numpy())
    )
    return complex_structure_factors


def complex_array_to_rs_dataseries(
    complex_structure_factors: np.ndarray,
    index: pd.Index,
) -> tuple[rs.DataSeries, rs.DataSeries]:
    """
    Convert an array of complex structure factors into two reciprocalspaceship DataSeries, one
    representing the structure factor amplitudes and one for the phases.

    Parameters
    ----------
    complex_structure_factors: np.ndarray
        the complex-valued structure factors, as a numpy array
    index: pandas.Index
        the indices (HKL) for each structure factor in `complex_structure_factors`

    Returns
    -------
    amplitudes: DataSeries
        with StructureFactorAmplitudeDtype
    phases: DataSeries
        with PhaseDtype

    Raises
    ------
    ValueError
        if `complex_structure_factors and `index` do not have the same shape
    """
    if not complex_structure_factors.shape == index.shape:
        raise ShapeMismatchError(
            f"shape of `complex_structure_factors` ({complex_structure_factors.shape}) does not "
            f"match shape of `index` ({index.shape})"
        )

    amplitudes = rs.DataSeries(np.abs(complex_structure_factors), index=index)
    amplitudes = amplitudes.astype(rs.StructureFactorAmplitudeDtype())

    phases = rs.DataSeries(np.rad2deg(np.angle(complex_structure_factors)), index=index)
    phases = phases.astype(rs.PhaseDtype())

    return amplitudes, phases


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
