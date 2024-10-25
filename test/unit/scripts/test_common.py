from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.rsmap import Map
from meteor.scripts.common import DiffmapArgParser, DiffMapSet, WeightMode


def mocked_read_mtz(dummy_filename: Path) -> rs.DataSet:
    assert isinstance(dummy_filename, Path)
    index = pd.MultiIndex.from_arrays([[1, 1, 5], [1, 2, 5], [1, 3, 5]], names=("H", "K", "L"))
    data = {
        "F": np.array([2.0, 3.0, 1.0]),
        "SIGF": np.array([0.5, 0.5, 1.0]),
    }
    return rs.DataSet(data, index=index).infer_mtz_dtypes()


def test_diffmap_set_smoke(diffmap_set: DiffMapSet) -> None:
    assert isinstance(diffmap_set, DiffMapSet)


@pytest.mark.parametrize("use_uncertainties", [False, True])
def test_diffmap_set_scale(diffmap_set: DiffMapSet, use_uncertainties: bool) -> None:
    diffmap_set.calculated["F"] *= 2

    # upon scale, both native and derivative should also become 2x bigger
    native_amps_before = diffmap_set.native["F"].to_numpy()
    derivative_amps_before = diffmap_set.native["F"].to_numpy()

    diffmap_set.scale()

    assert np.all(native_amps_before * 2 == diffmap_set.native["F"].to_numpy())
    assert np.all(derivative_amps_before * 2 == diffmap_set.derivative["F"].to_numpy())


def test_diffmap_argparser_parse_args(base_cli_arguments: list[str]) -> None:
    parser = DiffmapArgParser()
    args = parser.parse_args(base_cli_arguments)

    assert args.derivative_mtz == Path("fake-derivative.mtz")
    assert args.derivative_amplitude_column == "F"
    assert args.derivative_uncertainty_column == "SIGF"
    assert args.native_mtz == Path("fake-native.mtz")
    assert args.native_amplitude_column == "infer"
    assert args.native_uncertainty_column == "infer"
    assert args.pdb == Path("fake.pdb")
    assert args.mtzout == Path("fake-output.mtz")
    assert args.metadataout == Path("fake-output-metadata.csv")
    assert args.kweight_mode == WeightMode.fixed
    assert args.kweight_parameter == 0.75


def test_diffmap_argparser_check_output_filepaths(
    base_cli_arguments: list[str], tmp_path: Path
) -> None:
    parser = DiffmapArgParser()
    args = parser.parse_args(base_cli_arguments)

    # this should pass; no files on disk
    parser.check_output_filepaths(args)

    existing = tmp_path / "exists.foo"
    existing.open("a").close()

    args.mtzout = existing
    with pytest.raises(IOError, match="file: "):
        parser.check_output_filepaths(args)

    args.mtzout = Path("fine-output-filename.mtz")
    parser.check_output_filepaths(args)

    args.metadataout = existing
    with pytest.raises(IOError, match="file: "):
        parser.check_output_filepaths(args)


@mock.patch("meteor.scripts.common.rs.read_mtz", mocked_read_mtz)
def test_contruct_map() -> None:
    # map phases have an extra index
    calculated_map_phases = rs.DataSeries([60.0, 181.0, -91.0, 0.0])
    index = pd.MultiIndex.from_arrays(
        [[1, 1, 5, 6], [1, 2, 5, 6], [1, 3, 5, 7]], names=("H", "K", "L")
    )
    calculated_map_phases.index = index

    constructed_map = DiffmapArgParser._construct_map(
        name="fake-name",
        mtz_file=Path("function-is-mocked.mtz"),
        calculated_map_phases=calculated_map_phases,
        amplitude_column="F",
        uncertainty_column="SIGF",
    )
    assert len(constructed_map) == 3
    assert constructed_map.has_uncertainties

    constructed_map = DiffmapArgParser._construct_map(
        name="fake-name",
        mtz_file=Path("function-is-mocked.mtz"),
        calculated_map_phases=calculated_map_phases,
        amplitude_column="infer",
        uncertainty_column="infer",
    )
    assert len(constructed_map) == 3
    assert constructed_map.has_uncertainties


def test_load_difference_maps(random_difference_map: Map, base_cli_arguments: list[str]) -> None:
    parser = DiffmapArgParser()
    args = parser.parse_args(base_cli_arguments)

    def return_a_map(*args: Any, **kwargs: Any) -> Map:
        return random_difference_map

    mocked_fxn_1 = "meteor.scripts.common.structure_to_calculated_map"
    mocked_fxn_2 = "meteor.scripts.common.DiffmapArgParser._construct_map"

    with mock.patch(mocked_fxn_1, return_a_map), mock.patch(mocked_fxn_2, return_a_map):
        mapset = DiffmapArgParser.load_difference_maps(args)
        assert isinstance(mapset.native, Map)
        assert isinstance(mapset.derivative, Map)
        assert isinstance(mapset.calculated, Map)
