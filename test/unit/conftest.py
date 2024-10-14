from __future__ import annotations

import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs

from meteor.utils import (
    MapColumns,
    canonicalize_amplitudes,
    compute_coefficients_from_map,
    numpy_array_to_map,
)
from meteor.rsmap import Map

RESOLUTION = 1.0
UNIT_CELL = gemmi.UnitCell(a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90)
SPACE_GROUP = gemmi.find_spacegroup_by_name("P1")
CARBON1_POSITION = (5.0, 5.0, 5.0)
CARBON2_POSITION = (5.0, 5.2, 5.0)

NP_RNG = np.random.default_rng()


@pytest.fixture
def test_map_columns() -> MapColumns:
    return MapColumns(
        amplitude="F",
        phase="PHIC",
        uncertainty="SIGF",
    )


@pytest.fixture
def test_diffmap_columns() -> MapColumns:
    return MapColumns(
        amplitude="DF",
        phase="PHIC",
        uncertainty="SIGDF",
    )


def single_carbon_density(
    carbon_position: tuple[float, float, float],
    space_group: gemmi.SpaceGroup,
    unit_cell: gemmi.UnitCell,
    d_min: float,
) -> gemmi.FloatGrid:
    model = gemmi.Model("single_atom")
    chain = gemmi.Chain("A")

    residue = gemmi.Residue()
    residue.name = "X"
    residue.seqid = gemmi.SeqId("1")

    atom = gemmi.Atom()
    atom.name = "C"
    atom.element = gemmi.Element("C")
    atom.pos = gemmi.Position(*carbon_position)

    residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)

    structure = gemmi.Structure()
    structure.add_model(model)
    structure.cell = unit_cell
    structure.spacegroup_hm = space_group.hm

    density_map = gemmi.DensityCalculatorX()
    density_map.d_min = d_min
    density_map.grid.setup_from(structure)
    density_map.put_model_density_on_grid(structure[0])

    return density_map.grid


def single_atom_map_coefficients(*, noise_sigma: float, labels: MapColumns) -> Map:
    density = np.array(single_carbon_density(CARBON1_POSITION, SPACE_GROUP, UNIT_CELL, RESOLUTION))
    grid_values = np.array(density) + noise_sigma * NP_RNG.normal(size=density.shape)
    ccp4_map = numpy_array_to_map(grid_values, spacegroup=SPACE_GROUP, cell=UNIT_CELL)

    # map_coefficients = compute_coefficients_from_map(
    #     ccp4_map=ccp4_map,
    #     high_resolution_limit=RESOLUTION,
    #     amplitude_label=labels.amplitude,
    #     phase_label=labels.phase,
    # )

    map_coefficients = Map.from_gemmi(ccp4_map=ccp4_map, high_resolution_limit=RESOLUTION)

    uncertainties = noise_sigma * np.ones_like(map_coefficients.phases)
    uncertainties = rs.DataSeries(uncertainties, index=map_coefficients.index)
    uncertainties = uncertainties.astype(rs.StandardDeviationDtype())
    
    map_coefficients.add_uncertainties(uncertainties)

    return map_coefficients


# @pytest.fixture
# def noise_free_map(test_map_columns: MapColumns) -> rs.DataSet:
#     return single_atom_map_coefficients(noise_sigma=0.0, labels=test_map_columns)

@pytest.fixture
def noise_free_map(test_map_columns: MapColumns) -> rs.DataSet:
    return single_atom_map_coefficients(noise_sigma=0.0, labels=test_map_columns)


@pytest.fixture
def noisy_map(test_map_columns: MapColumns) -> rs.DataSet:
    return single_atom_map_coefficients(noise_sigma=0.03, labels=test_map_columns)


@pytest.fixture
def very_noisy_map(test_map_columns: MapColumns) -> rs.DataSet:
    return single_atom_map_coefficients(noise_sigma=1.0, labels=test_map_columns)


@pytest.fixture
def random_difference_map(test_diffmap_columns: MapColumns) -> rs.DataSet:
    resolution = 1.0
    cell = gemmi.UnitCell(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    space_group = gemmi.SpaceGroup(1)
    hall = rs.utils.generate_reciprocal_asu(cell, space_group, resolution, anomalous=False)
    sigma = 1.0

    h, k, l = hall.T  # noqa: E741
    number_of_reflections = len(h)

    ds = rs.DataSet(
        {
            "H": h,
            "K": k,
            "L": l,
            test_diffmap_columns.amplitude: sigma * NP_RNG.normal(size=number_of_reflections),
            test_diffmap_columns.phase: NP_RNG.uniform(-180, 180, size=number_of_reflections),
        },
        spacegroup=space_group,
        cell=cell,
    ).infer_mtz_dtypes()

    ds = ds.set_index(["H", "K", "L"])
    ds[test_diffmap_columns.amplitude] = ds[test_diffmap_columns.amplitude].astype("SFAmplitude")

    canonicalize_amplitudes(
        ds,
        amplitude_label=test_diffmap_columns.amplitude,
        phase_label=test_diffmap_columns.phase,
        inplace=True,
    )

    uncertainties = sigma * np.ones_like(ds[test_diffmap_columns.amplitude])
    uncertainties = rs.DataSeries(uncertainties, index=ds.index)
    ds[test_diffmap_columns.uncertainty] = uncertainties.astype(rs.StandardDeviationDtype())

    return ds
