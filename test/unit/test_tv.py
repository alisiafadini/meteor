from typing import Sequence

import numpy as np
import pytest
import reciprocalspaceship as rs

from meteor import tv
from meteor.utils import MapLabels, compute_map_from_coefficients

LAMBDAS_TO_SCAN = np.logspace(-3, 1, 50)


def rms_between_coefficients(ds1: rs.DataSet, ds2: rs.DataSet, diffmap_labels: MapLabels, map_sampling: int = 3) -> float:
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

    # standardize -- TODO better to scale? think...
    map1_array /= map1_array.std()
    map2_array /= map2_array.std()

    rms = float(np.linalg.norm(map2_array - map1_array))

    return rms


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

# TODO re-implement None
@pytest.mark.parametrize("lambda_values_to_scan", [LAMBDAS_TO_SCAN,])
def test_tv_denoise_difference_map(
    lambda_values_to_scan: None | Sequence[float],
    noise_free_map: rs.DataSet,
    noisy_map: rs.DataSet,
    diffmap_labels: MapLabels,
) -> None:
    
    def rms_to_noise_free(test_map: rs.DataSet) -> float:
        return rms_between_coefficients(test_map, noise_free_map, diffmap_labels)

    # Normally, the `tv_denoise_difference_map` function only returns the best result -- since we 
    # know  the ground truth, work around this to test all possible results.
    
    lowest_rms: float = np.inf
    best_lambda: float = 0.0

    for trial_lambda in LAMBDAS_TO_SCAN:
        denoised_map = tv.tv_denoise_difference_map(
            difference_map_coefficients=noisy_map,
            lambda_values_to_scan=[trial_lambda,]
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
    np.testing.assert_approx_equal(result.optimal_lambda, best_lambda)
    np.testing.assert_approx_equal(rms_after_denoising, lowest_rms)
