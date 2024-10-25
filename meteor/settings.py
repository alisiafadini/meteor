from __future__ import annotations


TV_WEIGHT_RANGE: tuple[float, float] = (0.0, 1.0)
TV_STOP_TOLERANCE: float = 0.00000005
TV_MAX_NUM_ITER: int = 50
MAP_SAMPLING: int = 3

KWEIGHT_PARAMETER_DEFAULT: float = 0.05
TV_WEIGHT_DEFAULT: float = 0.01  # TODO: optimize

COMPUTED_MAP_RESOLUTION_LIMIT: float = 1.0
GEMMI_HIGH_RESOLUTION_BUFFER: float = 1e-6
