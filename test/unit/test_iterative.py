import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import reciprocalspaceship as rs

from meteor import iterative


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


# def test_iterative_tv(displaced_atom_two_datasets_noisy: rs.DataSet) -> None:
#     result = iterative.iterative_tv_phase_retrieval(displaced_atom_two_datasets_noisy)
#     for label in ["F", "Fh"]:
#         pdt.assert_series_equal(
#             result[label], displaced_atom_two_datasets_noisy[label], atol=1e-3
#         )
#     assert_phases_allclose(
#         result["PHIC"], displaced_atom_two_datasets_noisy["PHIC"], atol=1e-3
#     )
#     # assert_phases_allclose(
#     #     result["PHICh"], displaced_atom_two_datasets_noise_free["PHICh_ground_truth"], atol=1e-3
#     # )
