from __future__ import annotations

from typing import TYPE_CHECKING

import gemmi
import reciprocalspaceship as rs

from meteor.utils import (
    ShapeMismatchError,
    canonicalize_amplitudes,
    complex_array_to_rs_dataseries,
    rs_dataseries_to_complex_array,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd

GEMMI_HIGH_RESOLUTION_BUFFER = 1e-6


class Map(rs.DataSet):
    def __init__(
        self,
        *,
        amplitudes: rs.DataSeries,
        phases: rs.DataSeries,
        uncertainties: rs.DataSeries | None = None,
        amplitude_column: str = "F",
        phase_column: str = "PHI",
        uncertainty_column: str = "SIGF",
    ) -> None:

        inputs = [amplitudes, phases]
        if uncertainties:
            inputs.append(uncertainties)

        for input_series in inputs:
            if not hasattr(input_series, "index"):
                msg = f"no index set for {input_series.__name__}"
                raise AttributeError(msg)

        dataset = rs.concat(inputs, axis=1)
        if len(dataset) == 0:
            msg = "no common indices found in inputs"
            raise ValueError(msg)
        
        super().__init__(dataset)

        self._amplitude_column = amplitude_column
        self._phase_column = phase_column
        if uncertainties:
            self._uncertainty_column = uncertainty_column

        canonicalize_amplitudes(
            self,
            amplitude_label=self._amplitude_column,
            phase_label=self._phase_column,
            inplace=True,
        )

    def __setitem__(self, key: str, value) -> None:
        if key not in self.columns:
            msg = "only amplitude, phase, and uncertainty columns allowed; to add uncertainties "
            msg += "after object creation, see Map.add_uncertainties(...)"
            raise ValueError(msg)
        super().__setitem__(key, value)

    def insert(self, *args, **kwargs) -> None:
        msg = "only amplitude, phase, and uncertainty columns allowed; to add uncertainties "
        msg += "after object creation, see Map.add_uncertainties(...)"
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
    def uncertainty_column(self) -> str | None:
        if hasattr(self, "_uncertainty_column"):
            return self._uncertainty_column
        return None

    @uncertainty_column.setter
    def uncertainty_column(self, name: str) -> None:
        self.rename(columns={self.uncertainty_column: name}, inplace=True)
        self._uncertainty_column = name

    def add_uncertainties(self, uncertainties: rs.DataSeries, *, name: str = "SIGF") -> None:
        if self.uncertainty_column is not None:
            msg = f"uncertainty column {self.uncertainty_column} already assigned"
            raise KeyError(msg)
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
        if self.uncertainty_column is None:
            msg = "Map object has no uncertainties set, see Map.add_uncertainties(...)"
            raise KeyError(msg)
        return self[self.uncertainty_column]

    @property
    def complex_structure_factors(self) -> np.ndarray:
        return rs_dataseries_to_complex_array(amplitudes=self.amplitudes, phases=self.phases)

    @classmethod
    def to_gemmi(
        cls,
        *,
        map_sampling: int,
    ) -> gemmi.Ccp4Map:
        map_coefficients_gemmi_format = super().to_gemmi()
        ccp4_map = gemmi.Ccp4Map()
        ccp4_map.grid = map_coefficients_gemmi_format.transform_f_phi_to_map(
            cls.amplitude, cls.phase, sample_rate=map_sampling
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
    ):
        return cls(
            amplitudes=dataset[amplitude_column],
            phases=dataset[phase_column],
            uncertainties=(dataset[uncertainty_column] if uncertainty_column else None),
        )

    @classmethod
    def from_complex_sfs(
        cls,
        *,
        complex_structure_factors: np.ndarray,
        index: pd.Index,
        uncertainties: np.ndarray | None,
    ):
        amplitudes, phases = complex_array_to_rs_dataseries(
            complex_structure_factors=complex_structure_factors, index=index
        )
        if uncertainties:
            if complex_structure_factors.shape != uncertainties.shape:
                msg = "`complex_sfs` and `uncertainties` do not have same shapes"
                raise ShapeMismatchError(msg)
            return cls(amplitudes=amplitudes, phases=phases, uncertainties=uncertainties)
        return cls(amplitudes=amplitudes, phases=phases)

    @classmethod
    def from_gemmi(
        cls,
        *,
        ccp4_map: gemmi.Ccp4Map,
        high_resolution_limit: float,
        amplitude_column: str = "F",
        phase_column: str = "PHI",
    ):
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
        #dataset = super().from_gemmi(mtz)
        dataset = rs.DataSet.from_gemmi(mtz)

        return cls.from_dataset(dataset, amplitude_column=amplitude_column, phase_column=phase_column)

    @classmethod
    def from_mtz_file(
        cls,
        file_path: Path,
        *,
        amplitude_column: str,
        phase_column: str,
        uncertainty_column: str,
    ):
        dataset = super().from_mtz_file(file_path)
        return cls.from_dataset(
            dataset,
            amplitude_column=amplitude_column,
            phase_column=phase_column,
            uncertainty_column=uncertainty_column,
        )
