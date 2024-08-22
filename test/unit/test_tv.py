from typing import Sequence

import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs

from meteor import tv
from meteor.utils import MapLabels, compute_coefficients_from_map, compute_map_from_coefficients


def _generate_single_carbon_density(
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


def displaced_single_atom_difference_map_coefficients(
    *,
    noise_sigma: float,
) -> rs.DataSet:
    unit_cell = gemmi.UnitCell(a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90)
    space_group = gemmi.find_spacegroup_by_name("P1")
    d_min = 1.0

    carbon_position1 = (5.0, 5.0, 5.0)
    carbon_position2 = (5.1, 5.0, 5.0)

    density1 = _generate_single_carbon_density(carbon_position1, space_group, unit_cell, d_min)
    density2 = _generate_single_carbon_density(carbon_position2, space_group, unit_cell, d_min)

    ccp4_map = gemmi.Ccp4Map()
    grid_values = (
        np.array(density2) - np.array(density1) + noise_sigma * np.random.randn(*density2.shape)
    )
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


def rms_between_coefficients(ds1: rs.DataSet, ds2: rs.DataSet, diffmap_labels: MapLabels) -> float:
    map1 = compute_map_from_coefficients(
        map_coefficients=ds1,
        amplitude_label=diffmap_labels.amplitude,
        phase_label=diffmap_labels.phases,
        map_sampling=3,
    )
    map2 = compute_map_from_coefficients(
        map_coefficients=ds2,
        amplitude_label=diffmap_labels.amplitude,
        phase_label=diffmap_labels.phases,
        map_sampling=3,
    )

    map1_array = np.array(map2.grid)
    map2_array = np.array(map1.grid)

    map1_array /= map1_array.std()
    map2_array /= map2_array.std()

    rms = float(np.linalg.norm(map2_array - map1_array))

    return rms


@pytest.fixture()
def noise_free_map() -> rs.DataSet:
    return displaced_single_atom_difference_map_coefficients(noise_sigma=0.0)


@pytest.fixture()
def noisy_map() -> rs.DataSet:
    return displaced_single_atom_difference_map_coefficients(noise_sigma=0.03)


@pytest.mark.parametrize("lambda_values_to_scan", [None, np.logspace(-3, 2, 100)])
def test_tv_denoise_difference_map(
    lambda_values_to_scan: None | Sequence[float],
    noise_free_map: rs.DataSet,
    noisy_map: rs.DataSet,
    diffmap_labels: MapLabels,
) -> None:
    rms_before_denoising = rms_between_coefficients(noise_free_map, noisy_map, diffmap_labels)
    denoised_map, result = tv.tv_denoise_difference_map(
        difference_map_coefficients=noisy_map,
        lambda_values_to_scan=lambda_values_to_scan,
        full_output=True,
    )
    rms_after_denoising = rms_between_coefficients(noise_free_map, denoised_map, diffmap_labels)
    assert rms_after_denoising < rms_before_denoising

    # print("xyz", result.optimal_lambda, rms_before_denoising, rms_after_denoising)

    # testmap = compute_map_from_coefficients(
    #     map_coefficients=noise_free_map,
    #     amplitude_label=diffmap_labels.amplitude,
    #     phase_label=diffmap_labels.phases,
    #     map_sampling=1,
    # )
    # testmap.write_ccp4_map("original.ccp4")
    # testmap = compute_map_from_coefficients(
    #     map_coefficients=noisy_map,
    #     amplitude_label=diffmap_labels.amplitude,
    #     phase_label=diffmap_labels.phases,
    #     map_sampling=1,
    # )
    # testmap.write_ccp4_map("noisy.ccp4")
    # testmap = compute_map_from_coefficients(
    #     map_coefficients=denoised_map,
    #     amplitude_label=diffmap_labels.amplitude,
    #     phase_label=diffmap_labels.phases,
    #     map_sampling=1,
    # )
    # testmap.write_ccp4_map("denoised.ccp4")
