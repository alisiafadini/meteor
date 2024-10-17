from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MapColumns:
    amplitude: str
    phase: str
    uncertainty: str | None = None


def assert_phases_allclose(array1: np.ndarray, array2: np.ndarray, atol=1e-3):
    diff = array2 - array1
    diff = (diff + 180) % 360 - 180
    absolute_difference = np.sum(np.abs(diff)) / float(np.prod(diff.shape))
    if not absolute_difference < atol:
        msg = f"per element diff {absolute_difference} > tolerance {atol}"
        raise AssertionError(msg)
