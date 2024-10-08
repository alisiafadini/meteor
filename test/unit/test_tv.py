from typing import Sequence

import numpy as np
import pandas.testing as pdt
import pytest
import reciprocalspaceship as rs

from meteor import tv
from meteor.utils import MapLabels, compute_map_from_coefficients

DEFAULT_LAMBDA_VALUES_TO_SCAN = np.logspace(-2, 0, 30)


def rms_between_coefficients(
    ds1: rs.DataSet, ds2: rs.DataSet, diffmap_labels: MapLabels, map_sampling: int = 3
) -> float:
    map1 = compute_map_from_coefficients(
        map_coefficients=ds1,
        amplitude_label=diffmap_labels.amplitude,
        phase_label=diffmap_labels.phase,
        map_sampling=map_sampling,
    )
    map2 = compute_map_from_coefficients(
        map_coefficients=ds2,
        amplitude_label=diffmap_labels.amplitude,
        phase_label=diffmap_labels.phase,
        map_sampling=map_sampling,
    )

    map1_array = np.array(map1.grid)
    map2_array = np.array(map2.grid)

    # standardize
    map1_array /= map1_array.std()
    map2_array /= map2_array.std()

    rms = float(np.linalg.norm(map2_array - map1_array))

    return rms


def assert_phases_allclose(array1: np.ndarray, array2: np.ndarray, atol=1e-3):
    diff = array2 - array1
    diff = (diff + 180) % 360 - 180
    absolute_difference = np.sum(np.abs(diff)) / float(np.prod(diff.shape))
    if not absolute_difference < atol:
        raise ValueError(f"per element diff {absolute_difference} > tolerance {atol}")


@pytest.mark.parametrize(
    "lambda_values_to_scan",
    [
        None,
        [
            0.01,
        ],
    ],
)
@pytest.mark.parametrize("full_output", [False, True])
def test_tv_denoise_difference_map_smoke(
    lambda_values_to_scan: None | Sequence[float],
    full_output: bool,
    noisy_map: rs.DataSet,
) -> None:
    output = tv.tv_denoise_difference_map(
        difference_map_coefficients=noisy_map,
        lambda_values_to_scan=lambda_values_to_scan,
        full_output=full_output,
    )  # type: ignore
    if full_output:
        assert len(output) == 2
        assert isinstance(output[0], rs.DataSet)
        assert isinstance(output[1], tv.TvDenoiseResult)
    else:
        assert isinstance(output, rs.DataSet)


@pytest.mark.parametrize("lambda_values_to_scan", [None, DEFAULT_LAMBDA_VALUES_TO_SCAN])
def test_tv_denoise_difference_map(
    lambda_values_to_scan: None | Sequence[float],
    noise_free_map: rs.DataSet,
    noisy_map: rs.DataSet,
    diffmap_labels: MapLabels,
) -> None:
    def rms_to_noise_free(test_map: rs.DataSet) -> float:
        return rms_between_coefficients(test_map, noise_free_map, diffmap_labels)

    # Normally, the `tv_denoise_difference_map` function only returns the best result -- since we
    # know the ground truth, work around this to test all possible results.

    lowest_rms: float = np.inf
    best_lambda: float = 0.0

    for trial_lambda in DEFAULT_LAMBDA_VALUES_TO_SCAN:
        denoised_map, result = tv.tv_denoise_difference_map(
            difference_map_coefficients=noisy_map,
            lambda_values_to_scan=[
                trial_lambda,
            ],
            full_output=True,
        )
        rms = rms_to_noise_free(denoised_map)
        if rms < lowest_rms:
            lowest_rms = rms
            best_lambda = trial_lambda

    # now run the denoising algorithm and make sure we get a result that's close
    # to the one that minimizes the RMS error to the ground truth
    denoised_map, result = tv.tv_denoise_difference_map(
        difference_map_coefficients=noisy_map,
        lambda_values_to_scan=lambda_values_to_scan,
        full_output=True,
    )

    rms_after_denoising = rms_to_noise_free(denoised_map)
    assert rms_after_denoising < rms_to_noise_free(noisy_map)
    np.testing.assert_allclose(result.optimal_lambda, best_lambda, rtol=0.2)


def test_dataseries_l1_norm() -> None:
    series1 = rs.DataSeries([0, 0, 0], index=[0, 1, 2])
    series2 = rs.DataSeries([1, 1, 1], index=[0, 1, 3])
    norm = tv._dataseries_l1_norm(series1, series2)
    assert norm == 1.0


def test_dataseries_l1_norm_no_overlapping_indices() -> None:
    series1 = rs.DataSeries([0, 0, 0], index=[0, 1, 2])
    series2 = rs.DataSeries([1, 1, 1], index=[3, 4, 5])
    with pytest.raises(RuntimeError):
        tv._dataseries_l1_norm(series1, series2)


def test_projected_derivative_phase_identical_phases() -> None:
    hkls = [0, 1, 2]
    phases = rs.DataSeries([0.0, 30.0, 60.0], index=hkls)
    amplitudes = rs.DataSeries([1.0, 1.0, 1.0], index=hkls)

    derivative_phases = tv._projected_derivative_phase(
        difference_amplitudes=amplitudes,
        difference_phases=phases,
        native_amplitudes=amplitudes,
        native_phases=phases,
    )
    assert_phases_allclose(phases.to_numpy(), derivative_phases.to_numpy())


def test_projected_derivative_phase_opposite_phases() -> None:
    hkls = [0, 1, 2]
    native_phases = rs.DataSeries([0.0, 30.0, 60.0], index=hkls)

    # if DF = 0, then derivative and native phase should be the same
    derivative_phases = tv._projected_derivative_phase(
        difference_amplitudes=rs.DataSeries([0.0, 0.0, 0.0], index=hkls),
        difference_phases=native_phases,
        native_amplitudes=rs.DataSeries([1.0, 1.0, 1.0], index=hkls),
        native_phases=native_phases,
    )
    np.testing.assert_almost_equal(derivative_phases.to_numpy(), native_phases.to_numpy())


def test_iterative_tv(displaced_atom_two_datasets_noise_free: rs.DataSet) -> None:
    result = tv.iterative_tv_phase_retrieval(displaced_atom_two_datasets_noise_free)
    for label in ["F", "Fh"]:
        pdt.assert_series_equal(
            result[label], displaced_atom_two_datasets_noise_free[label], atol=1e-3
        )
    assert_phases_allclose(
        result["PHIC"], displaced_atom_two_datasets_noise_free["PHIC"], atol=1e-3
    )
    # assert_phases_allclose(
    #     result["PHICh"], displaced_atom_two_datasets_noise_free["PHICh_ground_truth"], atol=1e-3
    # )
    
