import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs

from meteor.utils import compute_coefficients_from_map

DEFAULT_RESOLUTION = 1.0
DEFAULT_CELL_SIZE = 10.0
DEFAULT_CARBON1_POSITION = (5.0, 5.0, 5.0)
DEFAULT_CARBON2_POSITION = (5.0, 5.2, 5.0)


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


def denity_to_noisy_coefficients(
    *,
    density: gemmi.FloatGrid | np.ndarray,
    noise_sigma: float,
    resolution: float,
    unit_cell: gemmi.UnitCell,
    space_group: gemmi.SpaceGroup,
) -> rs.DataSet:
    ccp4_map = gemmi.Ccp4Map()
    grid_values = np.array(density) + noise_sigma * np.random.randn(*density.shape)
    ccp4_map.grid = gemmi.FloatGrid(grid_values.astype(np.float32), unit_cell, space_group)
    ccp4_map.update_ccp4_header()

    difference_map_coefficients = compute_coefficients_from_map(
        ccp4_map=ccp4_map,
        high_resolution_limit=resolution,
        amplitude_label="DF",
        phase_label="PHIC",
    )
    assert (difference_map_coefficients.max() > 0.0).any()

    return difference_map_coefficients


def displaced_single_atom_difference_map_coefficients(
    *,
    noise_sigma: float,
) -> rs.DataSet:
    space_group = gemmi.find_spacegroup_by_name("P1")
    unit_cell = gemmi.UnitCell(
        a=DEFAULT_CELL_SIZE, b=DEFAULT_CELL_SIZE, c=DEFAULT_CELL_SIZE, alpha=90, beta=90, gamma=90
    )

    density1 = single_carbon_density(
        DEFAULT_CARBON1_POSITION, space_group, unit_cell, DEFAULT_RESOLUTION
    )
    density2 = single_carbon_density(
        DEFAULT_CARBON2_POSITION, space_group, unit_cell, DEFAULT_RESOLUTION
    )

    density = np.array(density2) - np.array(density1)

    return denity_to_noisy_coefficients(
        density=density,
        noise_sigma=noise_sigma,
        resolution=DEFAULT_RESOLUTION,
        unit_cell=unit_cell,
        space_group=space_group,
    )


def two_datasets_atom_displacement(
    *,
    noise_sigma: float,
) -> rs.DataSet:
    unit_cell = gemmi.UnitCell(
        a=DEFAULT_CELL_SIZE, b=DEFAULT_CELL_SIZE, c=DEFAULT_CELL_SIZE, alpha=90, beta=90, gamma=90
    )
    space_group = gemmi.find_spacegroup_by_name("P1")

    density1 = single_carbon_density(
        DEFAULT_CARBON1_POSITION, space_group, unit_cell, DEFAULT_RESOLUTION
    )
    density2 = single_carbon_density(
        DEFAULT_CARBON2_POSITION, space_group, unit_cell, DEFAULT_RESOLUTION
    )

    coefficents1 = denity_to_noisy_coefficients(
        density=density1,
        noise_sigma=noise_sigma,
        resolution=DEFAULT_RESOLUTION,
        unit_cell=unit_cell,
        space_group=space_group,
    ).rename(columns={"DF": "F"})

    coefficents2 = denity_to_noisy_coefficients(
        density=density2,
        noise_sigma=noise_sigma,
        resolution=DEFAULT_RESOLUTION,
        unit_cell=unit_cell,
        space_group=space_group,
    ).rename(columns={"DF": "Fh", "PHIC": "PHICh_ground_truth"})

    return rs.concat([coefficents1, coefficents2], axis=1)


@pytest.fixture
def noise_free_map() -> rs.DataSet:
    return displaced_single_atom_difference_map_coefficients(noise_sigma=0.0)


@pytest.fixture
def noisy_map() -> rs.DataSet:
    return displaced_single_atom_difference_map_coefficients(noise_sigma=0.03)


@pytest.fixture
def displaced_atom_two_datasets_noise_free() -> rs.DataSet:
    return two_datasets_atom_displacement(noise_sigma=0.0)


@pytest.fixture
def displaced_atom_two_datasets_noisy() -> rs.DataSet:
    return two_datasets_atom_displacement(noise_sigma=0.03)
