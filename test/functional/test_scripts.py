from pathlib import Path

import pytest
import reciprocalspaceship as rs
from numpy import testing as npt

from meteor.rsmap import Map
from meteor.scale import scale_maps


def check_test_file_exists(path: Path) -> None:
    if not path.exists():
        msg = f"cannot find {path}, use github LFS to retrieve this file from the parent repo"
        raise OSError(msg)


@pytest.fixture
def scaled_test_data_mtz(test_data_dir: Path) -> Path:
    path = test_data_dir / "scaled-test-data.mtz"
    check_test_file_exists(path)
    return path


def test_scaling_regression(scaled_test_data_mtz: Path) -> None:
    ds = rs.read_mtz(str(scaled_test_data_mtz))

    on = Map(ds, amplitude_column="F_on", phase_column="PHIC", uncertainty_column="SIGF_on")
    off = Map(ds, amplitude_column="F_off", phase_column="PHIC", uncertainty_column="SIGF_off")
    calculated = Map(ds, amplitude_column="FC", phase_column="PHIC")

    scaled_on_truth = Map(
        ds, amplitude_column="scaled_on", phase_column="PHIC", uncertainty_column="SIGF_on_s"
    )
    scaled_off_truth = Map(
        ds, amplitude_column="scaled_off", phase_column="PHIC", uncertainty_column="SIGF_off_s"
    )

    scaled_on = scale_maps(
        map_to_scale=on, reference_map=calculated, weight_using_uncertainties=False
    )
    scaled_off = scale_maps(
        map_to_scale=off, reference_map=calculated, weight_using_uncertainties=False
    )

    npt.assert_allclose(scaled_on.amplitudes, scaled_on_truth.amplitudes, atol=1e-3)
    npt.assert_allclose(scaled_off.amplitudes, scaled_off_truth.amplitudes, atol=1e-3)
