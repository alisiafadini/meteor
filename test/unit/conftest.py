import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs

from meteor.utils import compute_coefficients_from_map

DEFAULT_RESOLUTION = 1.0
DEFAULT_CELL_SIZE = 10.0
DEFAULT_CARBON1_POSITION = (5.0, 5.0, 5.0)
DEFAULT_CARBON2_POSITION = (5.0, 5.2, 5.0)


def generate_single_carbon_density(
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


@pytest.fixture
def displaced_single_atom_difference_map_coefficients(
    *,
    noise_sigma: float,
    d_min: float = DEFAULT_RESOLUTION,
    cell_size: float = DEFAULT_CELL_SIZE,
    carbon_position1: tuple[float, float, float] = DEFAULT_CARBON1_POSITION,
    carbon_position2: tuple[float, float, float] = DEFAULT_CARBON2_POSITION,
) -> rs.DataSet:
    unit_cell = gemmi.UnitCell(a=cell_size, b=cell_size, c=cell_size, alpha=90, beta=90, gamma=90)
    space_group = gemmi.find_spacegroup_by_name("P1")

    density1 = generate_single_carbon_density(carbon_position1, space_group, unit_cell, d_min)
    density2 = generate_single_carbon_density(carbon_position2, space_group, unit_cell, d_min)

    ccp4_map = gemmi.Ccp4Map()
    grid_values = (
        np.array(density2) - np.array(density1) + noise_sigma * np.random.randn(*density2.shape)
    )
    ccp4_map.grid = gemmi.FloatGrid(grid_values.astype(np.float32), unit_cell, space_group)
    ccp4_map.update_ccp4_header()

    difference_map_coefficients = compute_coefficients_from_map(
        ccp4_map=ccp4_map,
        high_resolution_limit=d_min,
        amplitude_label="DF",
        phase_label="PHIC",
    )
    assert (difference_map_coefficients.max() > 0.0).any()

    return difference_map_coefficients


@pytest.fixture
def noise_free_map() -> rs.DataSet:
    return displaced_single_atom_difference_map_coefficients(noise_sigma=0.0)


@pytest.fixture
def noisy_map() -> rs.DataSet:
    return displaced_single_atom_difference_map_coefficients(noise_sigma=0.03)
