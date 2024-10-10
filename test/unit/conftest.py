import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs

from meteor.utils import compute_coefficients_from_map, numpy_array_to_map

RESOLUTION = 1.0
UNIT_CELL = gemmi.UnitCell(a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90)
SPACE_GROUP = gemmi.find_spacegroup_by_name("P1")
CARBON1_POSITION = (5.0, 5.0, 5.0)
CARBON2_POSITION = (5.0, 5.2, 5.0)


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


def carbon1_density() -> gemmi.FloatGrid:
    return single_carbon_density(CARBON1_POSITION, SPACE_GROUP, UNIT_CELL, RESOLUTION)


def carbon2_density() -> gemmi.FloatGrid:
    return single_carbon_density(CARBON2_POSITION, SPACE_GROUP, UNIT_CELL, RESOLUTION)


def displaced_single_atom_difference_map_coefficients(
    *,
    noise_sigma: float,
) -> rs.DataSet:
    difference_density = np.array(carbon1_density()) - np.array(carbon2_density())
    grid_values = np.array(difference_density) + noise_sigma * np.random.randn(
        *difference_density.shape
    )

    ccp4_map = numpy_array_to_map(grid_values, spacegroup=SPACE_GROUP, cell=UNIT_CELL)

    difference_map_coefficients = compute_coefficients_from_map(
        ccp4_map=ccp4_map,
        high_resolution_limit=RESOLUTION,
        amplitude_label="DF",
        phase_label="PHIC",
    )

    return difference_map_coefficients


@pytest.fixture
def carbon_difference_density() -> np.ndarray:
    difference_density = np.array(carbon1_density()) - np.array(carbon2_density())
    return difference_density


@pytest.fixture
def noise_free_map() -> rs.DataSet:
    return displaced_single_atom_difference_map_coefficients(noise_sigma=0.0)


@pytest.fixture
def noisy_map() -> rs.DataSet:
    return displaced_single_atom_difference_map_coefficients(noise_sigma=0.03)
