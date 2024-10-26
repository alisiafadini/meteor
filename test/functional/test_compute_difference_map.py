from pathlib import Path

from meteor.scripts import compute_difference_map
from meteor.scripts.common import WeightMode


def test_script_produces_consistent_results(
    testing_pdb_file: Path,
    testing_mtz_file: Path,
    tmp_path: Path,
) -> None:
    kweight_mode = WeightMode.fixed
    kweight_parameter = 0.05

    tv_weight_mode = WeightMode.fixed
    tv_weight = 0.1

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
        "--pdb",
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
