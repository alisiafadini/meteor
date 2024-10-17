from __future__ import annotations

from typing import Literal, overload

import gemmi
import numpy as np
import reciprocalspaceship as rs
from pandas import DataFrame, Index
from reciprocalspaceship.utils import canonicalize_phases


class ShapeMismatchError(Exception): ...


def filter_common_indices(df1: DataFrame, df2: DataFrame) -> tuple[DataFrame, DataFrame]:
    common_indices = df1.index.intersection(df2.index)
    df1_common = df1.loc[common_indices].copy()
    df2_common = df2.loc[common_indices].copy()
    if len(df1_common) == 0 or len(df2_common) == 0:
        msg = "cannot find any HKL incdices in common between `df1` and `df2`"
        raise IndexError(msg)
    return df1_common, df2_common


def cut_resolution(
    dataset: rs.DataSet,
    *,
    dmax_limit: float | None = None,
    dmin_limit: float | None = None,
) -> rs.DataSet:
    d_hkl = dataset.compute_dHKL()
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


def complex_array_to_rs_dataseries(
    complex_structure_factors: np.ndarray,
    *,
    index: Index,
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

    See Also
    --------
    `reciprocalspaceship/utils/structurefactors.from_structurefactor(...)`
        An equivalent function, that does not require the index and does less index/data
        checking.
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
