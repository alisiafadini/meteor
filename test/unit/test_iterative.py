import gemmi
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import reciprocalspaceship as rs
from skimage.data import binary_blobs
from skimage.restoration import denoise_tv_chambolle

from meteor import iterative
from meteor.testing import assert_phases_allclose
from meteor.tv import TvDenoiseResult
from meteor.utils import compute_map_from_coefficients


def map_norm(map1: gemmi.Ccp4Map, map2: gemmi.Ccp4Map) -> float:
    diff = np.array(map1.grid) - np.array(map2.grid)
    return float(np.linalg.norm(diff))


def simple_tv_function(fourier_array: np.ndarray) -> tuple[np.ndarray, TvDenoiseResult]:
    weight = 0.0001
    real_space = np.fft.ifftn(fourier_array).real
    denoised = denoise_tv_chambolle(real_space, weight=weight)
    result = TvDenoiseResult(
        optimal_lambda=weight,
        optimal_negentropy=0.0,
        lambdas_scanned=set([weight]),
        map_sampling_used_for_tv=0,
    )
    return np.fft.fftn(denoised), result


def normalized_rms(x: np.ndarray, y: np.ndarray) -> float:
    normalized_x = x / x.mean()
    normalized_y = y / y.mean()
    return float(np.linalg.norm(normalized_x - normalized_y))


@pytest.mark.parametrize("scalar", [0.01, 1.0, 2.0, 100.0])
def test_projected_derivative(scalar: float) -> None:
    n = 16
    native = np.random.randn(n) + 1j * np.random.randn(n)
    derivative = np.random.randn(n) + 1j * np.random.randn(n)
    difference = derivative - native

    # ensure the projection removes a scalar multiple of the native & difference
    scaled_native = scalar * native
    scaled_difference = scalar * difference
    proj_derivative = iterative._project_derivative_on_experimental_set(
        native=scaled_native, derivative_amplitudes=np.abs(derivative), difference=scaled_difference
    )
    np.testing.assert_allclose(proj_derivative, derivative)


def test_complex_derivative_from_iterative_tv() -> None:
    test_image = binary_blobs(length=64)

    constant_image = np.ones_like(test_image) / 2.0
    constant_image_ft = np.fft.fftn(constant_image)

    test_image_noisy = test_image + 0.2 * np.random.randn(*test_image.shape)
    test_image_noisy_ft = np.fft.fftn(test_image_noisy)

    denoised_derivative, _ = iterative._complex_derivative_from_iterative_tv(
        native=constant_image_ft,
        initial_derivative=test_image_noisy_ft,
        tv_denoise_function=simple_tv_function,
    )

    denoised_test_image = np.fft.ifftn(denoised_derivative).real

    noisy_error = normalized_rms(denoised_test_image, test_image)
    denoised_error = normalized_rms(test_image_noisy, test_image)

    # insist on 5% improvement
    assert 1.05 * noisy_error < denoised_error


def test_iterative_tv(single_atom_maps_noisy_and_noise_free: rs.DataSet) -> None:
    # the test case is the denoising of a difference: between a noisy map and its noise-free origin
    # such a diffmap is ideally totally flat, so should have very low TV
    

    result, metadata = iterative.iterative_tv_phase_retrieval(
        single_atom_maps_noisy_and_noise_free,
        native_amplitude_column="F_noise_free",
        calculated_phase_column="PHIC_noise_free",
        derivative_amplitude_column="F_noisy",
        output_derivative_phase_column="PHIC_denoised",
        tv_weights_to_scan=[0.01, 0.1, 1.0],
        max_iterations=100,
        convergence_tolerance=0.01,
    )

    # make sure output columns that should not be altered are in fact the same
    assert_phases_allclose(
        result["PHIC_noise_free"], single_atom_maps_noisy_and_noise_free["PHIC_noise_free"]
    )
    for label in ["F_noise_free", "F_noisy"]:
        pdt.assert_series_equal(result[label], single_atom_maps_noisy_and_noise_free[label])

    # make sure metadata exists
    assert isinstance(metadata, pd.DataFrame)
    print(metadata)

    # test correctness by comparing denoised dataset to noise-free
    map_sampling = 3
    noise_free_density = compute_map_from_coefficients(
        map_coefficients=single_atom_maps_noisy_and_noise_free,
        amplitude_label="F_noise_free",
        phase_label="PHIC_noise_free",
        map_sampling=map_sampling,
    )
    noisy_density = compute_map_from_coefficients(
        map_coefficients=single_atom_maps_noisy_and_noise_free,
        amplitude_label="F_noisy",
        phase_label="PHIC_noisy",
        map_sampling=map_sampling,
    )
    denoised_density = compute_map_from_coefficients(
        map_coefficients=result,
        amplitude_label="F_noisy",
        phase_label="PHIC_denoised",
        map_sampling=map_sampling,
    )
    noisy_error = map_norm(noisy_density, noise_free_density)
    denoised_error = map_norm(denoised_density, noise_free_density)

    # insist on 1% or better improvement
    assert 1.01 * denoised_error < noisy_error
