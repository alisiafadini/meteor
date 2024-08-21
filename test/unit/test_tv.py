import reciprocalspaceship as rs
from meteor.utils import compute_coefficients_from_map, compute_map_from_coefficients
from meteor import tv
import gemmi
import numpy as np
from pytest import fixture

# TODO make these universal in the tests
TEST_AMPLITUDE_LABEL = "DF"
TEST_PHASE_LABEL = "PHIC"


def _generate_single_carbon_density(
    carbon_position: tuple[float, float, float],
    space_group: gemmi.SpaceGroup,
    unit_cell: gemmi.UnitCell,
    d_min: float,
) -> gemmi.FloatGrid:
    # Create a model with a single carbon atom
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


def displaced_single_atom_difference_map_coefficients(
    *, noise_sigma: float,
) -> rs.DataSet:
    unit_cell = gemmi.UnitCell(a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90)
    space_group = gemmi.find_spacegroup_by_name("P1")
    d_min = 1.0

    carbon_position1 = (5.0, 5.0, 5.0)
    carbon_position2 = (5.1, 5.0, 5.0)

    density1 = _generate_single_carbon_density(carbon_position1, space_group, unit_cell, d_min)
    density2 = _generate_single_carbon_density(carbon_position2, space_group, unit_cell, d_min)

    ccp4_map = gemmi.Ccp4Map()
    grid_values = np.array(density2) - np.array(density1) + noise_sigma * np.random.randn(*density2.shape)
    ccp4_map.grid = gemmi.FloatGrid(grid_values.astype(np.float32), unit_cell, space_group)
    ccp4_map.update_ccp4_header()

    difference_map_coefficients = compute_coefficients_from_map(
        ccp4_map=ccp4_map,
        high_resolution_limit=1.0,
        amplitude_label="DF",
        phase_label="PHIC",
    )
    assert (difference_map_coefficients.max() > 0.0).any()

    return difference_map_coefficients


def rms_between_coefficients(ds1: rs.DataSet, ds2: rs.DataSet) -> float:
    map1 = compute_map_from_coefficients(
        map_coefficients=ds1,
        amplitude_label=TEST_AMPLITUDE_LABEL,
        phase_label=TEST_PHASE_LABEL,
        map_sampling=3
    )
    map2 = compute_map_from_coefficients(
        map_coefficients=ds2,
        amplitude_label=TEST_AMPLITUDE_LABEL,
        phase_label=TEST_PHASE_LABEL,
        map_sampling=3
    )
    difference_map = np.array(map2.grid) - np.array(map1.grid)
    rms = float(np.linalg.norm(difference_map))
    return rms


@fixture
def noise_free_map() -> rs.DataSet:
    return displaced_single_atom_difference_map_coefficients(noise_sigma=0.0)


@fixture
def noisy_map() -> rs.DataSet:
    return displaced_single_atom_difference_map_coefficients(noise_sigma=3.0)


def test_tv_denoise_difference_map_smoke(flat_difference_map: rs.DataSet) -> None:
    # test sequence pf specified lambda
    tv.tv_denoise_difference_map(
        difference_map_coefficients=flat_difference_map,
        lambda_values_to_scan=[1.0, 2.0],
    )
    # test golden optimizer
    tv.TV_LAMBDA_RANGE = (1.0, 2.0)
    tv.tv_denoise_difference_map(
        difference_map_coefficients=flat_difference_map,
    )


def test_tv_denoise_difference_map_golden(noise_free_map: rs.DataSet, noisy_map: rs.DataSet) -> None:
    rms_before_denoising = rms_between_coefficients(noise_free_map, noisy_map)
    denoised_map = tv.tv_denoise_difference_map(
        difference_map_coefficients=noisy_map,
    )
    rms_after_denoising = rms_between_coefficients(noise_free_map, denoised_map)
    assert rms_after_denoising < rms_before_denoising


def test_tv_denoise_difference_map_specific_lambdas(noise_free_map: rs.DataSet, noisy_map: rs.DataSet) -> None:
    rms_before_denoising = rms_between_coefficients(noise_free_map, noisy_map)
    denoised_map = tv.tv_denoise_difference_map(
        difference_map_coefficients=noisy_map,
        lambda_values_to_scan=np.logspace(-3, 0, 25),
    )
    rms_after_denoising = rms_between_coefficients(noise_free_map, denoised_map)
    assert rms_after_denoising < rms_before_denoising
