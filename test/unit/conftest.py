from __future__ import annotations

import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs

from meteor.rsmap import Map
from meteor.testing import MapColumns, single_carbon_density
from meteor.utils import numpy_array_to_map

RESOLUTION = 1.0
UNIT_CELL = gemmi.UnitCell(a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90)
SPACE_GROUP = gemmi.find_spacegroup_by_name("P1")
CARBON1_POSITION = (5.0, 5.0, 5.0)

NP_RNG = np.random.default_rng()

@pytest.fixture
def tv_denoise_result_source_data() -> dict:
    return {
        "initial_negentropy": 0.0,
        "optimal_tv_weight": 1.0,
        "optimal_negentropy": 5.0,
        "map_sampling_used_for_tv": 5,
        "tv_weights_scanned": [0.0, 1.0],
        "negentropy_at_weights": [0.0, 5.0],
        "k_parameter_used": 0.0,
    }


@pytest.fixture
def test_map_columns() -> MapColumns:
    return MapColumns(
        amplitude="F",
        phase="PHI",
        uncertainty="SIGF",
    )


def single_atom_map_coefficients(*, noise_sigma: float) -> Map:
    density_map = single_carbon_density(CARBON1_POSITION, SPACE_GROUP, UNIT_CELL, RESOLUTION)
    density_array = np.array(density_map.grid)
    grid_values = density_array + noise_sigma * NP_RNG.normal(size=density_array.shape)
    ccp4_map = numpy_array_to_map(grid_values, spacegroup=SPACE_GROUP, cell=UNIT_CELL)

    map_coefficients = Map.from_ccp4_map(ccp4_map=ccp4_map, high_resolution_limit=RESOLUTION)

    uncertainties = noise_sigma * np.ones_like(map_coefficients.phases)
    uncertainties = rs.DataSeries(uncertainties, index=map_coefficients.index)
    map_coefficients.set_uncertainties(uncertainties)

    return map_coefficients


@pytest.fixture
def ccp4_map() -> gemmi.Ccp4Map:
    return single_carbon_density(CARBON1_POSITION, SPACE_GROUP, UNIT_CELL, RESOLUTION)


@pytest.fixture
def noise_free_map() -> Map:
    return single_atom_map_coefficients(noise_sigma=0.0)


@pytest.fixture
def noisy_map() -> Map:
    return single_atom_map_coefficients(noise_sigma=0.03)


@pytest.fixture
def very_noisy_map() -> Map:
    return single_atom_map_coefficients(noise_sigma=1.0)


@pytest.fixture
def random_difference_map(test_map_columns: MapColumns) -> Map:
    hall = rs.utils.generate_reciprocal_asu(UNIT_CELL, SPACE_GROUP, RESOLUTION, anomalous=False)
    sigma = 1.0

    h, k, l = hall.T  # noqa: E741
    number_of_reflections = len(h)

    ds = rs.DataSet(
        {
            "H": h,
            "K": k,
            "L": l,
            test_map_columns.amplitude: sigma * NP_RNG.normal(size=number_of_reflections),
            test_map_columns.phase: NP_RNG.uniform(-180, 180, size=number_of_reflections),
        },
        spacegroup=SPACE_GROUP,
        cell=UNIT_CELL,
    ).infer_mtz_dtypes()

    ds = ds.set_index(["H", "K", "L"])
    ds[test_map_columns.amplitude] = ds[test_map_columns.amplitude].astype("SFAmplitude")

    uncertainties = sigma * np.ones_like(ds[test_map_columns.amplitude])
    uncertainties = rs.DataSeries(uncertainties, index=ds.index)
    ds[test_map_columns.uncertainty] = uncertainties.astype(rs.StandardDeviationDtype())

    return Map(
        ds,
        amplitude_column=test_map_columns.amplitude,
        phase_column=test_map_columns.phase,
        uncertainty_column=test_map_columns.uncertainty,
    )
