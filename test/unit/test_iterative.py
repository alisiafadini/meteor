from __future__ import annotations

import gemmi
import numpy as np
import pandas as pd
import pytest
from skimage.data import binary_blobs
from skimage.restoration import denoise_tv_chambolle

from meteor.iterative import (
    _complex_derivative_from_iterative_tv,
    _project_derivative_on_experimental_set,
    iterative_tv_phase_retrieval,
)
from meteor.rsmap import Map
from meteor.tv import TvDenoiseResult
from meteor.validate import negentropy


def map_norm(map1: gemmi.Ccp4Map, map2: gemmi.Ccp4Map) -> float:
    diff = np.array(map1.grid) - np.array(map2.grid)
    return float(np.linalg.norm(diff))


def simple_tv_function(fourier_array: np.ndarray) -> tuple[np.ndarray, TvDenoiseResult]:
    weight = 0.0001
    real_space = np.fft.ifftn(fourier_array).real
    denoised = denoise_tv_chambolle(real_space, weight=weight)  # type: ignore[no-untyped-call]
    result = TvDenoiseResult(
        initial_negentropy=0.0,
        optimal_tv_weight=weight,
        optimal_negentropy=1.0,
        tv_weights_scanned=[weight],
        negentropy_at_weights=[1.0],
        map_sampling_used_for_tv=0,
    )
    return np.fft.fftn(denoised), result


def normalized_rms(x: np.ndarray, y: np.ndarray) -> float:
    normalized_x = x / x.mean()
    normalized_y = y / y.mean()
    return float(np.linalg.norm(normalized_x - normalized_y))


@pytest.mark.parametrize("scalar", [0.01, 1.0, 2.0, 100.0])
def test_projected_derivative(scalar: float, np_rng: np.random.Generator) -> None:
    n = 16
    native = np_rng.normal(size=n) + 1j * np_rng.normal(size=n)
    derivative = np_rng.normal(size=n) + 1j * np_rng.normal(size=n)
    difference = derivative - native

    # ensure the projection removes a scalar multiple of the native & difference
    scaled_native = scalar * native
    scaled_difference = scalar * difference
    proj_derivative = _project_derivative_on_experimental_set(
        native=scaled_native,
        derivative_amplitudes=np.abs(derivative),
        difference=scaled_difference,
    )
    np.testing.assert_allclose(proj_derivative, derivative)


def test_complex_derivative_from_iterative_tv(np_rng: np.random.Generator) -> None:
    test_image = binary_blobs(length=64)  # type: ignore[no-untyped-call]

    constant_image = np.ones_like(test_image) / 2.0
    constant_image_ft = np.fft.fftn(constant_image)

    test_image_noisy = test_image + 0.2 * np_rng.normal(size=test_image.shape)
    test_image_noisy_ft = np.fft.fftn(test_image_noisy)

    denoised_derivative, _ = _complex_derivative_from_iterative_tv(
        native=constant_image_ft,
        initial_derivative=test_image_noisy_ft,
        tv_denoise_function=simple_tv_function,
        convergence_tolerance=0.001,
        max_iterations=1000,
    )

    denoised_test_image = np.fft.ifftn(denoised_derivative).real

    noisy_error = normalized_rms(denoised_test_image, test_image)
    denoised_error = normalized_rms(test_image_noisy, test_image)

    # insist on 5% improvement
    assert 1.05 * noisy_error < denoised_error


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
    map_sampling = 3
    noise_free_density = noise_free_map.to_ccp4_map(map_sampling=map_sampling)
    noisy_density = very_noisy_map.to_ccp4_map(map_sampling=3)
    denoised_density = denoised_map.to_ccp4_map(map_sampling=3)

    noisy_error = map_norm(noisy_density, noise_free_density)
    denoised_error = map_norm(denoised_density, noise_free_density)

    # insist on 1% or better improvement
    assert 1.01 * denoised_error < noisy_error

    # insist that the negentropy improves after denoising
    assert negentropy(np.array(denoised_density.grid)) > negentropy(np.array(noisy_density.grid))
