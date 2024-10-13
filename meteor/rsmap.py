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
    ) -> None:
        # enforce defaults, renaming allowed -- keeps column names in sync
        self._amplitude_column = "F"
        self._phase_column = "PHI"
        self._uncertainty_column = "SIGF"

        inputs = [amplitudes.rename(self._amplitude_column), phases.rename(self._phase_column)]
        if uncertainties:
            inputs.append(uncertainties.rename(self._uncertainty_column))

        for input_series in inputs:
            if not hasattr(input_series, "index"):
                msg = f"no index set for {input_series.__name__}"
                raise AttributeError(msg)

        dataset = rs.concat(inputs, axis=1)
        if len(dataset) == 0:
            msg = "no common indices found in inputs"
            raise ValueError(msg)

        canonicalize_amplitudes(
            dataset,
            amplitude_label=self._amplitude_column,
            phase_label=self._phase_column,
            inplace=True,
        )

        super().__init__(dataset)

    def __setitem__(self, key, value):
        if key not in self.columns:
            msg = "only amplitude, phase, and uncertainty columns allowed"
            raise ValueError(msg)
        super().__setitem__(key, value)

    @property
    def amplitude_column(self) -> str:
        return self._amplitude_column

    @amplitude_column.setter
    def amplitude_column(self, name: str) -> None:
        self.rename(columns={self._amplitude_column: name})
        self._amplitude_column = name

    @property
    def phase_column(self) -> str:
        return self._phase_column

    @phase_column.setter
    def phase_column(self, name: str) -> None:
        self.rename(columns={self.phase_column: name})
        self.phase_column = name

    @property
    def uncertainty_column(self) -> str:
        return self._uncertainty_column

    @uncertainty_column.setter
    def uncertainty_column(self, name: str) -> None:
        self.rename(columns={self.uncertainty_column: name})
        self.uncertainty_column = name

    @property
    def amplitudes(self) -> rs.DataSeries:
        return self[self.amplitude_column]

    @property
    def phases(self) -> rs.DataSeries:
        return self[self.phase_column]

    @property
    def uncertainties(self) -> rs.DataSeries:
        if self.uncertainty_column is None:
            msg = "Map object has no set uncertainties"
            raise KeyError(msg)
        return self[self.uncertainty_column]

    @property
    def complex_structure_factors(self) -> np.ndarray:
        return rs_dataseries_to_complex_array(amplitudes=self.amplitudes, phases=self.phases)

    @classmethod
    def to_realspace_map(
        cls,
        *,
        map_sampling: int,
    ) -> gemmi.Ccp4Map:
        map_coefficients_gemmi_format = cls.to_gemmi()
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
        uncertainty_column: str | None,
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
    def from_realspace_map(
        cls,
        *,
        ccp4_map: gemmi.Ccp4Map,
        high_resolution_limit: float,
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

        # TODO: audit below
        mtz.add_column(cls.amplitude, "F")
        mtz.add_column(cls.phase, "PHI")

        mtz.set_data(data)
        mtz.switch_to_asu_hkl()

        return rs.DataSet.from_gemmi(mtz)

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
