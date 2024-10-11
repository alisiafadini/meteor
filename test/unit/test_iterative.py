import numpy as np
import pandas.testing as pdt
import pytest
import reciprocalspaceship as rs
from skimage.data import binary_blobs
from skimage.restoration import denoise_tv_chambolle
from gemmi import Ccp4Map
import pandas as pd

from meteor import iterative
from meteor.testing import assert_phases_allclose
from meteor.tv import TvDenoiseResult
from meteor.utils import compute_map_from_coefficients


def normalized_map_std(map: Ccp4Map) -> float:
    map_array = np.array(map.grid)
    map_array /= map_array.mean()
    return np.std(map_array)


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


def test_iterative_tv(atom_and_noisy_atom: rs.DataSet) -> None:
    # TODO this test is still shit
    result, metadata = iterative.iterative_tv_phase_retrieval(
        atom_and_noisy_atom,
        tv_weights_to_scan=[0.001, 0.01, 0.1],
        max_iterations=100,
        convergence_tolerance=0.05,
    )

    # make sure output columns that should not be altered are in fact the same
    assert_phases_allclose(result["PHIC"], atom_and_noisy_atom["PHIC"], atol=1e-3)
    for label in ["F", "Fh"]:
        pdt.assert_series_equal(result[label], atom_and_noisy_atom[label], atol=1e-3)

    # make sure metadata exists
    assert isinstance(metadata, pd.DataFrame)

    noisy_density = compute_map_from_coefficients(
        map_coefficients=atom_and_noisy_atom, amplitude_label="Fh", phase_label="PHICh", map_sampling=3
    )
    denoised_density = compute_map_from_coefficients(
        map_coefficients=result, amplitude_label="Fh", phase_label="PHICh", map_sampling=3
    )
    print(metadata)
    assert normalized_map_std(denoised_density) < normalized_map_std(noisy_density)
