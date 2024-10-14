from __future__ import annotations

from typing import TYPE_CHECKING, Type

import gemmi
import numpy as np
import pandas as pd
import reciprocalspaceship as rs

from meteor.utils import (
    canonicalize_amplitudes,
    complex_array_to_rs_dataseries,
    rs_dataseries_to_complex_array,
)

if TYPE_CHECKING:
    from pathlib import Path


GEMMI_HIGH_RESOLUTION_BUFFER = 1e-6

CellType = Type[tuple[float, float, float, float, float, float] | gemmi.UnitCell]
SpaceGroupType = Type[int | str | gemmi.SpaceGroup]


class Map(rs.DataSet):
    def __init__(
        self,
        *,
        amplitudes: rs.DataSeries,
        phases: rs.DataSeries,
        uncertainties: rs.DataSeries | None = None,
        cell: CellType | None = None,
        spacegroup: SpaceGroupType | None = None,
        amplitude_column: str = "F",
        phase_column: str = "PHI",
        uncertainty_column: str = "SIGF",
    ) -> None:
        inputs = [amplitudes, phases]
        if uncertainties is not None:
            inputs.append(uncertainties)

        for input_series in inputs:
            if not hasattr(input_series, "index"):
                msg = f"no index set for {input_series.__name__}"
                raise AttributeError(msg)

        rs_dataset = rs.concat(inputs, axis=1)
        rs_dataset.cell = cell
        rs_dataset.spacegroup = spacegroup
        super().__init__(rs_dataset)

        self._amplitude_column = amplitude_column
        self._phase_column = phase_column
        if uncertainties is not None:
            self._uncertainty_column = uncertainty_column

        self._check_types()

        canonicalize_amplitudes(
            self,
            amplitude_label=amplitude_column,
            phase_label=phase_column,
            inplace=True,
        )

    # TODO: ugly
    def _check_types(self) -> None:
        amplitude_types = [rs.StructureFactorAmplitudeDtype, rs.FriedelStructureFactorAmplitudeDtype, rs.NormalizedStructureFactorAmplitudeDtype, rs.AnomalousDifferenceDtype]
        phase_types = [rs.PhaseDtype]
        uncertainty_types = [rs.StandardDeviationDtype, rs.StandardDeviationFriedelIDtype, rs.StandardDeviationFriedelSFDtype]
        
        if self[self.amplitude_column].dtype not in amplitude_types:
            self[self.amplitude_column] = self[self.amplitude_column].astype(rs.StructureFactorAmplitudeDtype())
        if self[self.phase_column].dtype not in phase_types:
            self[self.phase_column] = self[self.phase_column].astype(rs.PhaseDtype())
        if self.has_uncertainties:
            if self[self.uncertainty_column].dtype not in uncertainty_types:
                self[self.uncertainty_column] = self[self.uncertainty_column].astype(rs.StandardDeviationDtype())

    def __setitem__(self, key: str, value) -> None:
        if key not in self.columns:
            msg = "only amplitude, phase, and uncertainty columns allowed; to add uncertainties "
            msg += "after object creation, see Map.set_uncertainties(...)"
            raise KeyError(msg)
        super().__setitem__(key, value)

    def insert(self, *args, **kwargs) -> None:  # noqa: ARG002
        msg = "only amplitude, phase, and uncertainty columns allowed; to add uncertainties "
        msg += "after object creation, see Map.set_uncertainties(...)"
        raise NotImplementedError(msg)

    @property
    def amplitude_column(self) -> str:
        return self._amplitude_column

    @amplitude_column.setter
    def amplitude_column(self, name: str) -> None:
        self.rename(columns={self._amplitude_column: name}, inplace=True)
        self._amplitude_column = name

    @property
    def phase_column(self) -> str:
        return self._phase_column

    @phase_column.setter
    def phase_column(self, name: str) -> None:
        self.rename(columns={self.phase_column: name}, inplace=True)
        self._phase_column = name

    @property
    def has_uncertainties(self) -> bool:
        return hasattr(self, "_uncertainty_column")

    @property
    def uncertainty_column(self) -> str:
        if not self.has_uncertainties:
            msg = "uncertainty_column not set, no uncertainties"
            raise AttributeError(msg)
        return self._uncertainty_column

    @uncertainty_column.setter
    def uncertainty_column(self, name: str) -> None:
        self.rename(columns={self.uncertainty_column: name}, inplace=True)
        self._uncertainty_column = name

    def set_uncertainties(self, uncertainties: rs.DataSeries, *, name: str = "SIGF") -> None:
        if self.has_uncertainties:
            self.drop(self.uncertainty_column, axis=1, inplace=True)
        self._uncertainty_column = name
        position = len(self.columns)
        super().insert(position, name, uncertainties, allow_duplicates=False)

    @property
    def amplitudes(self) -> rs.DataSeries:
        return self[self.amplitude_column]

    @property
    def phases(self) -> rs.DataSeries:
        return self[self.phase_column]

    @property
    def uncertainties(self) -> rs.DataSeries:
        if not self.has_uncertainties:
            msg = "Map object has no uncertainties set, see Map.set_uncertainties(...)"
            raise KeyError(msg)
        return self[self.uncertainty_column]

    # TODO: naming of this function, relation to to_structurefactor
    @property
    def complex(self) -> np.ndarray:
        return rs_dataseries_to_complex_array(amplitudes=self.amplitudes, phases=self.phases)

    def to_structurefactor(self) -> rs.DataSeries:
        return super().to_structurefactor(self.amplitude_column, self.phase_column)

    def to_ccp4_map(self, *, map_sampling: int) -> gemmi.Ccp4Map:
        map_coefficients_gemmi_format = super().to_gemmi()
        ccp4_map = gemmi.Ccp4Map()
        ccp4_map.grid = map_coefficients_gemmi_format.transform_f_phi_to_map(
            self.amplitude_column, self.phase_column, sample_rate=map_sampling
        )
        ccp4_map.update_ccp4_header()
        return ccp4_map

    @classmethod
    def from_dataset(
        cls,
        dataset: rs.DataSet,
        *,
        amplitude_column: str,
        phase_column: str,
        uncertainty_column: str | None = None,
    ) -> Map:
        map_obj = cls(
            amplitudes=dataset[amplitude_column],
            phases=dataset[phase_column],
            uncertainties=(dataset[uncertainty_column] if uncertainty_column else None),
            spacegroup=dataset.spacegroup,
            cell=dataset.cell,
            amplitude_column=amplitude_column,
            phase_column=phase_column,
            uncertainty_column=(uncertainty_column if uncertainty_column else "None"),
        )
        return map_obj

    @classmethod
    def from_structurefactor(
        cls,
        complex_structurefactor: np.ndarray | rs.DataSeries,
        *,
        index: pd.Index | None = None,
        cell: CellType | None = None,
        spacegroup: SpaceGroupType | None = None,
    ) -> Map:
        if isinstance(complex_structurefactor, np.ndarray):
            if not isinstance(index, pd.Index):
                msg = "if `complex_structurefactor` is a numpy array, `index` must be provided"
                raise TypeError(msg)
            amplitudes, phases = complex_array_to_rs_dataseries(
                complex_structure_factors=complex_structurefactor, index=index
            )

        elif isinstance(complex_structurefactor, rs.DataSeries):
            amplitudes, phases = super().from_structurefactor(complex_structurefactor)

        else:
            msg = f"`complex_structurefactor` invalid type: {type(complex_structurefactor)}"
            raise TypeError(msg)

        return cls(amplitudes=amplitudes, phases=phases, spacegroup=spacegroup, cell=cell)

    @classmethod
    def from_ccp4_map(
        cls,
        ccp4_map: gemmi.Ccp4Map,
        *,
        high_resolution_limit: float,
        amplitude_column: str = "F",
        phase_column: str = "PHI",
    ) -> Map:
        # to ensure we include the final shell of reflections, add a small buffer to the resolution
        gemmi_structure_factors = gemmi.transform_map_to_f_phi(ccp4_map.grid, half_l=False)
        data = gemmi_structure_factors.prepare_asu_data(
            dmin=high_resolution_limit - GEMMI_HIGH_RESOLUTION_BUFFER, with_sys_abs=True
        )

        mtz = gemmi.Mtz(with_base=True)
        mtz.spacegroup = gemmi_structure_factors.spacegroup
        mtz.set_cell_for_all(gemmi_structure_factors.unit_cell)
        mtz.add_dataset("FromMap")

        mtz.add_column(amplitude_column, "F")
        mtz.add_column(phase_column, "P")

        mtz.set_data(data)
        mtz.switch_to_asu_hkl()

        # TODO: why doesnt this work?
        # dataset = super().from_gemmi(mtz)
        dataset = rs.DataSet.from_gemmi(mtz)

        return cls.from_dataset(
            dataset, amplitude_column=amplitude_column, phase_column=phase_column
        )

    @classmethod
    def from_mtz_file(
        cls,
        file_path: Path,
        *,
        amplitude_column: str,
        phase_column: str,
        uncertainty_column: str,
    ) -> Map:
        dataset = super().from_mtz_file(file_path)
        return cls.from_dataset(
            dataset,
            amplitude_column=amplitude_column,
            phase_column=phase_column,
            uncertainty_column=uncertainty_column,
        )

    # TODO: verify these OK
    # def from_records(): ...
    # def from_dict(): ...
    # def drop(): ...
