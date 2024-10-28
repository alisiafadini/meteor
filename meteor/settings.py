from __future__ import annotations

import numpy as np

# map computation
COMPUTED_MAP_RESOLUTION_LIMIT: float = 1.0
GEMMI_HIGH_RESOLUTION_BUFFER: float = 1e-6


# known map labels
OBSERVED_INTENSITY_COLUMNS: list[str] = [
    "I",  # generic
    "IMEAN",  # CCP4
    "I-obs",  # phenix
]
OBSERVED_AMPLITUDE_COLUMNS: list[str] = [
    "F",  # generic
    "FP",  # CCP4 & GLPh native
    r"FPH\d",  # CCP4 derivative
    "F-obs",  # phenix
]
OBSERVED_UNCERTAINTY_COLUMNS: list[str] = [
    "SIGF",  # generic
    "SIGFP",  # CCP4 & GLPh native
    r"SIGFPH\d",  # CCP4
]
COMPUTED_AMPLITUDE_COLUMNS: list[str] = ["FC"]
COMPUTED_PHASE_COLUMNS: list[str] = ["PHIC"]


# k-weighting
KWEIGHT_PARAMETER_DEFAULT: float = 0.05
DEFAULT_KPARAMS_TO_SCAN = np.linspace(0.0, 1.0, 101)


# tv denoising
TV_WEIGHT_DEFAULT: float = 0.01
BRACKET_FOR_GOLDEN_OPTIMIZATION: tuple[float, float] = (0.0, 0.05)
TV_STOP_TOLERANCE: float = 0.00000005
TV_MAX_NUM_ITER: int = 50
MAP_SAMPLING: int = 3

# iterative tv
DEFAULT_TV_WEIGHTS_TO_SCAN_AT_EACH_ITERATION = [0.001, 0.01, 0.1, 1.0]
