from __future__ import annotations

import gemmi
import numpy as np
import pandas as pd

from meteor.iterative import (
    iterative_tv_phase_retrieval,
)
from meteor.rsmap import Map
from meteor.testing import diffmap_realspace_rms
from meteor.validate import map_negentropy


def map_norm(map1: gemmi.Ccp4Map, map2: gemmi.Ccp4Map) -> float:
    diff = np.array(map1.grid) - np.array(map2.grid)
    return float(np.linalg.norm(diff))


def test_IterativeTvDenoiser() -> None:
    raise NotImplementedError

    # denoiser = _IterativeTvDenoiser(
    #     cell=noise_free_map.cell,
    #     spacegroup=noise_free_map.spacegroup,
    #     convergence_tolerance=0.001,
    #     max_iterations=1000,
    # )

    # denoised_derivative, _ = denoiser(
    #     native=noise_free_map.to_structurefactor(),
    #     initial_derivative=noisy_map.to_structurefactor(),
    # )

    # noisy_error = rms_between_coefficients(noisy_map, noise_free_map)
    # denoised_error = rms_between_coefficients(denoised_derivative, noise_free_map)

    # # insist on 5% improvement
    # assert 1.05 * noisy_error < denoised_error


def test_tv_denoise_complex_difference_sf() -> None:
    raise NotImplementedError


def test_iterative_tv_different_indices(noise_free_map: Map, very_noisy_map: Map) -> None:
    # regression test to make sure we can accept maps with different indices
    labels = pd.MultiIndex.from_arrays(
        [
            (1, 2),
        ]
        * 3,
        names=("H", "K", "L"),
    )

    n = len(very_noisy_map)
    very_noisy_map.drop(labels, inplace=True)
    assert len(very_noisy_map) == n - 2

    denoised_map, metadata = iterative_tv_phase_retrieval(
        very_noisy_map,
        noise_free_map,
        tv_weights_to_scan=[0.1],
        max_iterations=100,
        convergence_tolerance=0.01,
    )
    assert isinstance(metadata, pd.DataFrame)
    assert isinstance(denoised_map, Map)


def test_iterative_tv(noise_free_map: Map, very_noisy_map: Map) -> None:
    # the test case is the denoising of a difference: between a noisy map and its noise-free origin
    # such a diffmap is ideally totally flat, so should have very low TV

    denoised_map, metadata = iterative_tv_phase_retrieval(
        very_noisy_map,
        noise_free_map,
        tv_weights_to_scan=[0.01, 0.1, 1.0],
        max_iterations=100,
        convergence_tolerance=0.01,
    )

    # make sure metadata exists
    assert isinstance(metadata, pd.DataFrame)

    # test correctness by comparing denoised dataset to noise-free
    noisy_error = diffmap_realspace_rms(very_noisy_map, noise_free_map)
    denoised_error = diffmap_realspace_rms(denoised_map, noise_free_map)

    # insist on 1% or better improvement
    assert 1.01 * denoised_error < noisy_error

    # insist that the negentropy improves after denoising
    assert map_negentropy(denoised_map) > map_negentropy(very_noisy_map)
