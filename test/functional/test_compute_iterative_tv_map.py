from pathlib import Path

import numpy as np
import pytest
import reciprocalspaceship as rs

from meteor.rsmap import Map
from meteor.scripts import compute_iterative_tv_map
from meteor.scripts.common import WeightMode, read_combined_metadata
from meteor.utils import filter_common_indices


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
        "-x",
        "0.01",
        "--max-iterations",
        "5"
    ]

    compute_iterative_tv_map.main(cli_args)

    iterative_tv_metadata, final_tv_metadata = read_combined_metadata(filename=output_metadata)
    result_map = Map.read_mtz_file(output_mtz)

    # 1. make sure the negentropy increased during iterative TV
    negentropy_over_iterations = iterative_tv_metadata["negentropy_after_tv"].to_numpy()
    assert negentropy_over_iterations[-1] > negentropy_over_iterations[0]

    # 2. make sure negentropy increased in the final TV pass
    if kweight_mode == WeightMode.none and tv_weight_mode == WeightMode.none:
        np.testing.assert_allclose(
            final_tv_metadata.optimal_negentropy, final_tv_metadata.initial_negentropy
        )
    else:
        assert final_tv_metadata.optimal_negentropy >= final_tv_metadata.initial_negentropy

    # 3. make sure computed DF are close to those stored on disk
    reference_dataset = rs.read_mtz(str(testing_mtz_file))
    reference_amplitudes = reference_dataset["F_itTV"]

    result_amplitudes, reference_amplitudes = filter_common_indices(
        result_map.amplitudes, reference_amplitudes
    )
    rho = np.corrcoef(result_amplitudes.to_numpy(), reference_amplitudes.to_numpy())[0, 1]

    # comparing a correlation coefficienct allows for a global scale factor change, but nothing else
    if (kweight_mode == WeightMode.none) or (tv_weight_mode == WeightMode.none):  # noqa: PLR1714
        assert rho > 0.50
    else:
        assert rho > 0.95
