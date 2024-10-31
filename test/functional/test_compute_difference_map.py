from pathlib import Path

import numpy as np
import pytest
import reciprocalspaceship as rs

from meteor import settings
from meteor.rsmap import Map
from meteor.scripts import compute_difference_map
from meteor.scripts.common import WeightMode
from meteor.tv import TvDenoiseResult
from meteor.utils import filter_common_indices

settings.MAP_SAMPLING = 1  # faster tests


@pytest.mark.parametrize("kweight_mode", list(WeightMode))
@pytest.mark.parametrize("tv_weight_mode", list(WeightMode))
def test_script_produces_consistent_results(
    kweight_mode: WeightMode,
    tv_weight_mode: WeightMode,
    testing_pdb_file: Path,
    testing_mtz_file: Path,
    tmp_path: Path,
) -> None:
    # for when WeightMode.fixed; these maximize negentropy in manual testing
    kweight_parameter = 0.05
    tv_weight = 0.01

    output_mtz = tmp_path / "test-output.mtz"
    output_metadata = tmp_path / "test-output-metadata.csv"

    cli_args = [
        str(testing_mtz_file),  # derivative
        "--derivative-amplitude-column",
        "F_on",
        "--derivative-uncertainty-column",
        "SIGF_on",
        str(testing_mtz_file),  # native
        "--native-amplitude-column",
        "F_off",
        "--native-uncertainty-column",
        "SIGF_off",
        "--structure",
        str(testing_pdb_file),
        "-o",
        str(output_mtz),
        "-m",
        str(output_metadata),
        "--kweight-mode",
        kweight_mode,
        "--kweight-parameter",
        str(kweight_parameter),
        "--tv-denoise-mode",
        tv_weight_mode,
        "--tv-weight",
        str(tv_weight),
    ]

    compute_difference_map.main(cli_args)

    result_metadata = TvDenoiseResult.from_json_file(output_metadata)
    result_map = Map.read_mtz_file(output_mtz)

    # 1. make sure negentropy increased
    if kweight_mode == WeightMode.none and tv_weight_mode == WeightMode.none:
        np.testing.assert_allclose(
            result_metadata.optimal_negentropy, result_metadata.initial_negentropy
        )
    else:
        assert result_metadata.optimal_negentropy >= result_metadata.initial_negentropy

    # 2. make sure optimized weights close to expected
    if kweight_mode == WeightMode.optimize:
        assert result_metadata.k_parameter_used is not None, "optimized kparameter is None"
        np.testing.assert_allclose(
            kweight_parameter,
            result_metadata.k_parameter_used,
            err_msg="kweight optimium different from expected",
        )
    if tv_weight_mode == WeightMode.optimize:
        if kweight_mode == WeightMode.none:
            optimal_tv_no_kweighting = 0.025
            np.testing.assert_allclose(
                optimal_tv_no_kweighting,
                result_metadata.optimal_tv_weight,
                rtol=0.1,
                err_msg="tv weight optimium different from expected",
            )
        else:
            optimal_tv_with_weighting = 0.00867
            np.testing.assert_allclose(
                optimal_tv_with_weighting,
                result_metadata.optimal_tv_weight,
                rtol=0.1,
                err_msg="tv weight optimium different from expected",
            )

    # 3. make sure computed DF are close to those stored on disk
    reference_dataset = rs.read_mtz(str(testing_mtz_file))
    reference_amplitudes = reference_dataset["F_TV"]

    result_amplitudes, reference_amplitudes = filter_common_indices(
        result_map.amplitudes, reference_amplitudes
    )
    rho = np.corrcoef(result_amplitudes.to_numpy(), reference_amplitudes.to_numpy())[0, 1]

    # comparing a correlation coefficienct allows for a global scale factor change, but nothing else
    if (kweight_mode == WeightMode.none) or (tv_weight_mode == WeightMode.none):  # noqa: PLR1714
        assert rho > 0.50
    else:
        assert rho > 0.98
