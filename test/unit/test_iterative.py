import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import reciprocalspaceship as rs
from skimage.data import binary_blobs
from skimage.restoration import denoise_tv_chambolle

from meteor import iterative
from meteor.utils import assert_phases_allclose, compute_map_from_coefficients


def simple_tv_function(fourier_array: np.ndarray) -> np.ndarray:
    real_space = np.fft.ifftn(fourier_array).real
    denoised = denoise_tv_chambolle(real_space, weight=0.001)
    return np.fft.fftn(denoised)


def normalized_rms(x: np.ndarray, y: np.ndarray) -> float:
    normalized_x = x / x.mean()
    normalized_y = y / y.mean()
    return float(np.linalg.norm(normalized_x - normalized_y))


def test_l1_norm() -> None:
    n = 5
    x = np.arange(n)
    y = np.arange(n) + 1
    assert iterative._l1_norm(x, x) == 0.0
    assert iterative._l1_norm(x, y) == 1.0


def test_rs_dataseies_to_complex_array() -> None:
    index = pd.Index(np.arange(4))
    amp = rs.DataSeries(np.ones(4), index=index)
    phase = rs.DataSeries(np.arange(4) * 90.0, index=index)

    carray = iterative._rs_dataseies_to_complex_array(amp, phase)
    expected = np.array([1.0, 0.0, -1.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0, -1.0])

    np.testing.assert_almost_equal(carray, expected)


def test_complex_array_to_rs_dataseries() -> None:
    carray = np.array([1.0, 0.0, -1.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0, -1.0])
    index = pd.Index(np.arange(4))

    expected_amp = rs.DataSeries(np.ones(4), index=index).astype(rs.StructureFactorAmplitudeDtype())
    expected_phase = rs.DataSeries([0.0, 90.0, 180.0, -90.0], index=index).astype(rs.PhaseDtype())

    amp, phase = iterative._complex_array_to_rs_dataseries(carray, index)
    pdt.assert_series_equal(amp, expected_amp)
    pdt.assert_series_equal(phase, expected_phase)


def test_complex_array_dataseries_roundtrip() -> None:
    n = 5
    carray = np.random.randn(n) + 1j * np.random.randn(n)
    indices = pd.Index(np.arange(n))

    ds_amplitudes, ds_phases = iterative._complex_array_to_rs_dataseries(carray, indices)

    assert isinstance(ds_amplitudes, rs.DataSeries)
    assert isinstance(ds_phases, rs.DataSeries)

    assert ds_amplitudes.dtype == rs.StructureFactorAmplitudeDtype()
    assert ds_phases.dtype == rs.PhaseDtype()

    pdt.assert_index_equal(ds_amplitudes.index, indices)
    pdt.assert_index_equal(ds_phases.index, indices)

    carray2 = iterative._rs_dataseies_to_complex_array(ds_amplitudes, ds_phases)
    np.testing.assert_almost_equal(carray, carray2, decimal=5)


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

    denoised_derivative = iterative._complex_derivative_from_iterative_tv(
        native=constant_image_ft,
        initial_derivative=test_image_noisy_ft,
        tv_denoise_function=simple_tv_function,
    )

    denoised_test_image = np.fft.ifftn(denoised_derivative).real

    noisy_error = normalized_rms(denoised_test_image, test_image)
    denoised_error = normalized_rms(test_image_noisy, test_image)
    assert 1.05 * noisy_error < denoised_error


def test_iterative_tv(
    atom_minus_noisy_atom: rs.DataSet, carbon_difference_density: np.ndarray
) -> None:
    
    # get the initial diffmap
    atom_minus_noisy_atom["DF"] = (
        atom_minus_noisy_atom["Fh"] - atom_minus_noisy_atom["F"]
    )
    noisy_density = compute_map_from_coefficients(
        map_coefficients=atom_minus_noisy_atom,
        amplitude_label="DF",
        phase_label="PHIC",
        map_sampling=3,
    )

    # run it-TV
    result = iterative.iterative_tv_phase_retrieval(atom_minus_noisy_atom)

    # make sure output columns that should not be altered are in fact the same
    assert_phases_allclose(result["PHIC"], atom_minus_noisy_atom["PHIC"], atol=1e-3)
    for label in ["F", "Fh"]:
        pdt.assert_series_equal(result[label], atom_minus_noisy_atom[label], atol=1e-3)

    denoised_density = compute_map_from_coefficients(
        map_coefficients=result, amplitude_label="DF", phase_label="DPHI", map_sampling=3
    )

    noisy_error = normalized_rms(np.array(noisy_density.grid), carbon_difference_density)
    denoised_error = normalized_rms(np.array(denoised_density.grid), carbon_difference_density)
    #assert 1.01 * denoised_error < noisy_error

    # noisy_density.write_ccp4_map("noised.map")
    # denoised_density.write_ccp4_map("denoised.map")
