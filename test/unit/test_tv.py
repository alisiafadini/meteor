from typing import Sequence

import numpy as np
import pytest
import reciprocalspaceship as rs

from meteor import tv
from meteor.utils import MapLabels, compute_map_from_coefficients

DEFAULT_LAMBDA_VALUES_TO_SCAN = np.logspace(-2, 0, 25)


def rms_between_coefficients(
    ds1: rs.DataSet, ds2: rs.DataSet, labels: MapLabels, map_sampling: int = 3
) -> float:
    map1 = compute_map_from_coefficients(
        map_coefficients=ds1,
        amplitude_label=labels.amplitude,
        phase_label=labels.phase,
        map_sampling=map_sampling,
    )
    map2 = compute_map_from_coefficients(
        map_coefficients=ds2,
        amplitude_label=labels.amplitude,
        phase_label=labels.phase,
        map_sampling=map_sampling,
    )

    map1_array = np.array(map1.grid)
    map2_array = np.array(map2.grid)

    # standardize
    map1_array /= map1_array.std()
    map2_array /= map2_array.std()

    rms = float(np.linalg.norm(map2_array - map1_array))

    return rms


@pytest.mark.parametrize(
    "lambda_values_to_scan",
    [
        None,
        [0.01],
    ],
)
@pytest.mark.parametrize("full_output", [False, True])
def test_tv_denoise_map_smoke(
    lambda_values_to_scan: None | Sequence[float],
    full_output: bool,
    random_difference_map: rs.DataSet,
    test_diffmap_labels: MapLabels,
) -> None:
    output = tv.tv_denoise_difference_map(
        difference_map_coefficients=random_difference_map,
        lambda_values_to_scan=lambda_values_to_scan,
        full_output=full_output,
        difference_map_amplitude_column=test_diffmap_labels.amplitude,
        difference_map_phase_column=test_diffmap_labels.phase,
    )  # type: ignore
    if full_output:
        assert len(output) == 2
        assert isinstance(output[0], rs.DataSet)
        assert isinstance(output[1], tv.TvDenoiseResult)
    else:
        assert isinstance(output, rs.DataSet)


@pytest.mark.parametrize("lambda_values_to_scan", [None, DEFAULT_LAMBDA_VALUES_TO_SCAN])
def test_tv_denoise_map(
    lambda_values_to_scan: None | Sequence[float],
    noise_free_map: rs.DataSet,
    noisy_map: rs.DataSet,
    test_map_labels: MapLabels,
) -> None:
    def rms_to_noise_free(test_map: rs.DataSet) -> float:
        return rms_between_coefficients(test_map, noise_free_map, test_map_labels)

    # Normally, the `tv_denoise_difference_map` function only returns the best result -- since we
    # know the ground truth, work around this to test all possible results.

    lowest_rms: float = np.inf
    best_lambda: float = 0.0

    for trial_lambda in DEFAULT_LAMBDA_VALUES_TO_SCAN:
        denoised_map, result = tv.tv_denoise_difference_map(
            difference_map_coefficients=noisy_map,
            difference_map_amplitude_column=test_map_labels.amplitude,
            difference_map_phase_column=test_map_labels.phase,
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
        difference_map_amplitude_column=test_map_labels.amplitude,
        difference_map_phase_column=test_map_labels.phase,
        full_output=True,
    )

    assert rms_to_noise_free(denoised_map) < rms_to_noise_free(noisy_map), "error didnt drop"
    np.testing.assert_allclose(result.optimal_lambda, best_lambda, rtol=0.5, err_msg="opt lambda")
