from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

import gemmi
import numpy as np
import reciprocalspaceship as rs
from pandas.testing import assert_index_equal
from reciprocalspaceship.utils import canonicalize_phases

if TYPE_CHECKING:
    import pandas as pd

from .settings import GEMMI_HIGH_RESOLUTION_BUFFER


class ShapeMismatchError(Exception): ...


@dataclass
class MapColumns:
    amplitude: str
    phase: str
    uncertainty: str | None = None


def filter_common_indices(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_indices = df1.index.intersection(df2.index)
    df1_common = df1.loc[common_indices].copy()
    df2_common = df2.loc[common_indices].copy()
    return df1_common, df2_common


def cut_resolution(
    dataset: rs.DataSet,
    *,
    dmax_limit: float | None = None,
    dmin_limit: float | None = None,
) -> rs.DataSet:
    d_hkl = np.array(dataset.compute_dHKL())
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

    dataset[phase_label] = canonicalize_phases(dataset[phase_label])

    if not inplace:
        return dataset
    return None


def average_phase_diff_in_degrees(array1: np.ndarray, array2: np.ndarray) -> float:
    if array1.shape != array2.shape:
        msg = f"inputs not same shape: {array1.shape} vs {array2.shape}"
        raise ShapeMismatchError(msg)
    phase1 = np.rad2deg(np.angle(array1))
    phase2 = np.rad2deg(np.angle(array2))
    diff = phase2 - phase1
    diff = (diff + 180) % 360 - 180
    return np.sum(np.abs(diff)) / float(np.prod(array1.shape))


# TODO: the following two functions are duplicated in reciprocalspaceship
# https://github.com/rs-station/reciprocalspaceship/blob/ceae60e293bfdb3e969d0e3e2b53fa3a2b9e34f9/reciprocalspaceship/utils/structurefactors.py#L8
def rs_dataseries_to_complex_array(amplitudes: rs.DataSeries, phases: rs.DataSeries) -> np.ndarray:
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
        msg = (
            "Indices for `amplitudes` and `phases` don't match. To safely cast to a single complex",
            " array, pass DataSeries with a common set of indices. One possible way: ",
            "Series.align(other, join='inner', axis=0).",
        )
        raise ShapeMismatchError(msg) from exptn
    return amplitudes.to_numpy() * np.exp(1j * np.deg2rad(phases.to_numpy()))


def complex_array_to_rs_dataseries(
    complex_structure_factors: np.ndarray,
    *,
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
    if complex_structure_factors.shape != index.shape:
        msg = (
            f"shape of `complex_structure_factors` ({complex_structure_factors.shape}) does not "
            f"match shape of `index` ({index.shape})"
        )
        raise ShapeMismatchError(msg)

    amplitudes = rs.DataSeries(
        np.abs(complex_structure_factors),
        index=index,
        dtype=rs.StructureFactorAmplitudeDtype(),
        name="F",
    )

    phases = rs.DataSeries(
        np.angle(complex_structure_factors, deg=True),
        index=index,
        dtype=rs.PhaseDtype(),
        name="PHI",
    )

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


# TODO: do we need these two methods anymore? or can we just us rsmap.Map?
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
