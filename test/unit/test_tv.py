from typing import Sequence

import numpy as np
import pytest
import reciprocalspaceship as rs

from meteor import tv
from meteor.utils import MapLabels, compute_map_from_coefficients


def rms_between_coefficients(ds1: rs.DataSet, ds2: rs.DataSet, diffmap_labels: MapLabels) -> float:
    map1 = compute_map_from_coefficients(
        map_coefficients=ds1,
        amplitude_label=diffmap_labels.amplitude,
        phase_label=diffmap_labels.phase,
        map_sampling=3,
    )
    map2 = compute_map_from_coefficients(
        map_coefficients=ds2,
        amplitude_label=diffmap_labels.amplitude,
        phase_label=diffmap_labels.phase,
        map_sampling=3,
    )

    map1_array = np.array(map2.grid)
    map2_array = np.array(map1.grid)

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


@pytest.mark.parametrize("lambda_values_to_scan", [None, np.logspace(-3, 2, 100)])
def test_tv_denoise_difference_map(
    lambda_values_to_scan: None | Sequence[float],
    noise_free_map: rs.DataSet,
    noisy_map: rs.DataSet,
    diffmap_labels: MapLabels,
) -> None:
    rms_before_denoising = rms_between_coefficients(noise_free_map, noisy_map, diffmap_labels)
    denoised_map, result = tv.tv_denoise_difference_map(
        difference_map_coefficients=noisy_map,
        lambda_values_to_scan=lambda_values_to_scan,
        full_output=True,
    )
    rms_after_denoising = rms_between_coefficients(noise_free_map, denoised_map, diffmap_labels)
    assert rms_after_denoising < rms_before_denoising
