from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.iterative import (
    IterativeTvDenoiser,
    _assert_are_dataseries,
)
from meteor.rsmap import Map
from meteor.testing import diffmap_realspace_rms
from meteor.validate import map_negentropy


@pytest.fixture
def testing_denoiser() -> IterativeTvDenoiser:
    return IterativeTvDenoiser(
        tv_weights_to_scan=[0.1],
        convergence_tolerance=0.01,
        max_iterations=100,
    )


def test_assert_are_dataseries() -> None:
    ds = rs.DataSeries([1, -1, 1], index=[1, 2, 3])
    _assert_are_dataseries(ds)
    _assert_are_dataseries(ds, ds)

    with pytest.raises(TypeError):
        _assert_are_dataseries(1)  # type: ignore[arg-type]


def test_init(testing_denoiser: IterativeTvDenoiser) -> None:
    assert isinstance(testing_denoiser, IterativeTvDenoiser)


def test_tv_denoise_complex_difference_sf(
    testing_denoiser: IterativeTvDenoiser, np_rng: np.random.Generator
) -> None:
    # use a huge TV weight, make sure random noise goes down
    testing_denoiser.tv_weights_to_scan = [100.0]
    size = 256
    noise = rs.DataSeries(np_rng.normal(size) + 1j * np_rng.normal(size), index=np.arange(size))

    cell = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
    denoised_sfs, metadata = testing_denoiser._tv_denoise_complex_difference_sf(
        noise, cell=cell, spacegroup=1
    )

    assert isinstance(denoised_sfs, rs.DataSeries)
    assert isinstance(metadata, pd.DataFrame)

    assert np.sum(np.abs(denoised_sfs)) < np.sum(np.abs(noise))


def test_iteratively_denoise_sf_amplitudes() -> None: ...


def test_iterative_tv_denoiser_different_indices(
    noise_free_map: Map, very_noisy_map: Map, testing_denoiser: IterativeTvDenoiser
) -> None:
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

    denoised_map, metadata = testing_denoiser(derivative=very_noisy_map, native=noise_free_map)
    assert isinstance(metadata, pd.DataFrame)
    assert isinstance(denoised_map, Map)


def test_iterative_tv_denoiser(
    noise_free_map: Map, very_noisy_map: Map, testing_denoiser: IterativeTvDenoiser
) -> None:
    # the test case is the denoising of a difference: between a noisy map and its noise-free origin
    # such a diffmap is ideally totally flat, so should have very low TV

    denoised_map, metadata = testing_denoiser(derivative=very_noisy_map, native=noise_free_map)

    # make sure metadata exists
    assert isinstance(metadata, pd.DataFrame)
    for expected_col in ["iteration", "tv_weight", "negentropy_after_tv", "average_phase_change"]:
        assert expected_col in metadata.columns

    # test correctness by comparing denoised dataset to noise-free
    noisy_error = diffmap_realspace_rms(very_noisy_map, noise_free_map)
    denoised_error = diffmap_realspace_rms(denoised_map, noise_free_map)

    # insist on 1% or better improvement
    assert 1.01 * denoised_error < noisy_error

    # insist that the negentropy improves after denoising
    assert map_negentropy(denoised_map) > map_negentropy(very_noisy_map)
