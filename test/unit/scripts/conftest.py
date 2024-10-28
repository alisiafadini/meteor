from __future__ import annotations

import pytest

from meteor.rsmap import Map
from meteor.scripts.common import DiffMapSet


@pytest.fixture
def diffmap_set(random_difference_map: Map) -> DiffMapSet:
    return DiffMapSet(
        native=random_difference_map.copy(),
        derivative=random_difference_map.copy(),
        calculated=random_difference_map.copy(),
    )


@pytest.fixture
def fixed_kparameter() -> float:
    return 0.05


@pytest.fixture
def base_cli_arguments(fixed_kparameter: float) -> list[str]:
    return [
        "fake-derivative.mtz",
        "-da",
        "F",
        "--derivative-uncertainty-column",
        "SIGF",
        "fake-native.mtz",
        "--structure",
        "fake.pdb",
        "-o",
        "fake-output.mtz",
        "-m",
        "fake-output-metadata.csv",
        "--kweight-mode",
        "fixed",
        "--kweight-parameter",
        str(fixed_kparameter),
    ]
