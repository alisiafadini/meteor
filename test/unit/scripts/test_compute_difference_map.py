from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from meteor.rsmap import Map
from meteor.scripts import compute_difference_map
from meteor.scripts.common import DiffMapSet, WeightMode
from meteor.scripts.compute_difference_map import (
    TvDiffmapArgParser,
    denoise_diffmap,
    make_requested_diffmap,
)
from meteor.tv import TvDenoiseResult

# ensure tests complete quickly by monkey-patching a limited number of weights
compute_difference_map.TV_WEIGHTS_TO_SCAN = np.linspace(0.0, 0.1, 6)


@pytest.fixture
def tv_cli_arguments(base_cli_arguments: list[str]) -> list[str]:
    new_cli_arguments = [
        "-tv",
        "fixed",
        "--tv-weight",
        "0.1",
    ]
    return [*base_cli_arguments, *new_cli_arguments]


@pytest.fixture
def parsed_tv_cli_args(tv_cli_arguments: list[str]) -> argparse.Namespace:
    parser = TvDiffmapArgParser()
    return parser.parse_args(tv_cli_arguments)


def test_tv_diffmap_parser(parsed_tv_cli_args: argparse.Namespace) -> None:
    assert parsed_tv_cli_args.tv_denoise_mode == WeightMode.fixed
    assert parsed_tv_cli_args.tv_weight == 0.1


@pytest.mark.parametrize("mode", list(WeightMode))
def test_make_requested_diffmap(mode: WeightMode, diffmap_set: DiffMapSet) -> None:
    diffmap = make_requested_diffmap(mapset=diffmap_set, kweight_mode=mode, kweight_parameter=0.75)
    assert len(diffmap) > 0
    assert isinstance(diffmap, Map)

    if mode == WeightMode.fixed:
        with pytest.raises(TypeError):
            _ = make_requested_diffmap(
                mapset=diffmap_set, kweight_mode=mode, kweight_parameter=None
            )


@pytest.mark.parametrize("mode", list(WeightMode))
def test_denoise_diffmap(mode: WeightMode, random_difference_map: Map) -> None:
    diffmap, metadata = denoise_diffmap(
        diffmap=random_difference_map,
        tv_denoise_mode=mode,
        tv_weight=0.1,
    )

    assert len(diffmap) > 0
    assert isinstance(diffmap, Map)
    assert isinstance(metadata, TvDenoiseResult)

    if mode == WeightMode.optimize:
        assert np.isclose(metadata.optimal_weight, 0.06)
    elif mode == WeightMode.fixed:
        assert metadata.optimal_weight == 0.1
        with pytest.raises(TypeError):
            _, _ = denoise_diffmap(
                diffmap=random_difference_map, tv_denoise_mode=mode, tv_weight=None
            )
    elif mode == WeightMode.none:
        assert metadata.optimal_weight == 0.0


def test_main(diffmap_set: DiffMapSet, tmp_path: Path) -> None:
    def mock_load_difference_maps(self: Any, args: argparse.Namespace) -> DiffMapSet:
        return diffmap_set

    output_mtz_path = tmp_path / "out.mtz"
    output_metadata_path = tmp_path / "metadata.csv"

    cli_arguments = [
        "fake-derivative.mtz",
        "fake-native.mtz",
        "--pdb",
        "fake.pdb",
        "-o",
        str(output_mtz_path),
        "-m",
        str(output_metadata_path),
        "--kweight-mode",
        "fixed",
        "--kweight-parameter",
        "0.75",
        "-tv",
        "fixed",
        "-l",
        "0.1",
    ]

    fxn_to_mock = "meteor.scripts.compute_difference_map.TvDiffmapArgParser.load_difference_maps"
    with mock.patch(fxn_to_mock, mock_load_difference_maps):
        compute_difference_map.main(cli_arguments)

    assert output_mtz_path.exists()
    assert output_metadata_path.exists()
