from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from meteor.scripts import compute_iterative_tv_map
from meteor.scripts.common import DiffMapSet
from meteor.scripts.compute_iterative_tv_map import (
    IterativeTvArgParser,
)

TV_WEIGHTS_TO_SCAN = [0.01, 0.05]


@pytest.fixture
def tv_cli_arguments(base_cli_arguments: list[str]) -> list[str]:
    new_cli_arguments = [
        "--tv-weights-to-scan",
        *[str(weight) for weight in TV_WEIGHTS_TO_SCAN],
    ]
    return [*base_cli_arguments, *new_cli_arguments]


@pytest.fixture
def parsed_tv_cli_args(tv_cli_arguments: list[str]) -> argparse.Namespace:
    parser = IterativeTvArgParser()
    return parser.parse_args(tv_cli_arguments)


def test_tv_diffmap_parser(parsed_tv_cli_args: argparse.Namespace) -> None:
    assert parsed_tv_cli_args.tv_weights_to_scan == TV_WEIGHTS_TO_SCAN


def test_main(diffmap_set: DiffMapSet, tmp_path: Path, fixed_kparameter: float) -> None:
    def mock_load_difference_maps(self: Any, args: argparse.Namespace) -> DiffMapSet:
        return diffmap_set

    output_mtz_path = tmp_path / "out.mtz"
    output_metadata_path = tmp_path / "metadata.csv"

    cli_arguments = [
        "fake-derivative.mtz",
        "fake-native.mtz",
        "--structure",
        "fake.pdb",
        "-o",
        str(output_mtz_path),
        "-m",
        str(output_metadata_path),
        "--kweight-mode",
        "fixed",
        "--kweight-parameter",
        str(fixed_kparameter),
        "-x",
        *[str(weight) for weight in TV_WEIGHTS_TO_SCAN],
    ]

    # TODO is very slow

    fxn_to_mck = "meteor.scripts.compute_iterative_tv_map.IterativeTvArgParser.load_difference_maps"
    with mock.patch(fxn_to_mck, mock_load_difference_maps):
        compute_iterative_tv_map.main(cli_arguments)

    assert output_mtz_path.exists()
    assert output_metadata_path.exists()
