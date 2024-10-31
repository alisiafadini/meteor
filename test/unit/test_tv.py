from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from meteor import tv
from meteor.rsmap import Map

DEFAULT_WEIGHTS_TO_SCAN = np.logspace(-2, 0, 25)


def rms_between_coefficients(map1: Map, map2: Map) -> float:
    map_sampling = 3
    map1_array = np.array(map1.to_ccp4_map(map_sampling=map_sampling).grid)
    map2_array = np.array(map2.to_ccp4_map(map_sampling=map_sampling).grid)

    # standardize
    map1_array /= map1_array.std()
    map2_array /= map2_array.std()

    return float(np.linalg.norm(map2_array - map1_array))


def test_tv_denoise_result(tv_denoise_result_source_data: dict) -> None:
    tdr_obj = tv.TvDenoiseResult(**tv_denoise_result_source_data)
    assert tv_denoise_result_source_data == asdict(tdr_obj)

    json = tdr_obj.json()
    roundtrip = tv.TvDenoiseResult.from_json(json)
    assert tv_denoise_result_source_data == asdict(roundtrip)


def test_tv_denoise_result_to_file(tv_denoise_result_source_data: dict, tmp_path: Path) -> None:
    tdr_obj = tv.TvDenoiseResult(**tv_denoise_result_source_data)
    filepath = tmp_path / "tmp.json"
    tdr_obj.to_json_file(filepath)
    roundtrip = tv.TvDenoiseResult.from_json_file(filepath)
    assert tv_denoise_result_source_data == asdict(roundtrip)


@pytest.mark.parametrize(
    "weights_to_scan",
    [
        None,
        [-1.0, 0.0, 1.0],
    ],
)
@pytest.mark.parametrize("full_output", [False, True])
def test_tv_denoise_map_smoke(
    weights_to_scan: None | Sequence[float],
    full_output: bool,
    random_difference_map: Map,
) -> None:
    output = tv.tv_denoise_difference_map(
        random_difference_map,
        weights_to_scan=weights_to_scan,
        full_output=full_output,
    )  # type: ignore[call-overload]
    if full_output:
        assert len(output) == 2
        assert isinstance(output[0], Map)
        assert isinstance(output[1], tv.TvDenoiseResult)
    else:
        assert isinstance(output, Map)


def test_tv_denoise_zero_weight(random_difference_map: Map) -> None:
    weight = 0.0
    output = tv.tv_denoise_difference_map(
        random_difference_map,
        weights_to_scan=[weight],
        full_output=False,
    )
    random_difference_map.canonicalize_amplitudes()
    output.canonicalize_amplitudes()
    pd.testing.assert_frame_equal(random_difference_map, output, atol=1e-2, rtol=1e-2)


def test_tv_denoise_nan_input(random_difference_map: Map) -> None:
    weight = 0.0
    random_difference_map.iloc[0] = np.nan
    _ = tv.tv_denoise_difference_map(
        random_difference_map,
        weights_to_scan=[weight],
        full_output=False,
    )


@pytest.mark.parametrize("weights_to_scan", [None, DEFAULT_WEIGHTS_TO_SCAN])
def test_tv_denoise_map(
    weights_to_scan: None | Sequence[float],
    noise_free_map: Map,
    noisy_map: Map,
) -> None:
    def rms_to_noise_free(test_map: Map) -> float:
        return rms_between_coefficients(test_map, noise_free_map)

    # Normally, the `tv_denoise_difference_map` function only returns the best result -- since we
    # know the ground truth, work around this to test all possible results.

    lowest_rms: float = np.inf
    best_weight: float = 0.0

    for trial_weight in DEFAULT_WEIGHTS_TO_SCAN:
        denoised_map, result = tv.tv_denoise_difference_map(
            noisy_map,
            weights_to_scan=[
                trial_weight,
            ],
            full_output=True,
        )
        rms = rms_to_noise_free(denoised_map)
        if rms < lowest_rms:
            lowest_rms = rms
            best_weight = trial_weight

    # now run the denoising algorithm and make sure we get a result that's close
    # to the one that minimizes the RMS error to the ground truth
    denoised_map, result = tv.tv_denoise_difference_map(
        noisy_map,
        weights_to_scan=weights_to_scan,
        full_output=True,
    )

    assert rms_to_noise_free(denoised_map) < rms_to_noise_free(noisy_map), "error didnt drop"
    np.testing.assert_allclose(
        result.optimal_tv_weight, best_weight, rtol=0.5, err_msg="opt weight"
    )
