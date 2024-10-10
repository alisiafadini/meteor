import numpy as np


def assert_phases_allclose(array1: np.ndarray, array2: np.ndarray, atol=1e-3):
    diff = array2 - array1
    diff = (diff + 180) % 360 - 180
    absolute_difference = np.sum(np.abs(diff)) / float(np.prod(diff.shape))
    if not absolute_difference < atol:
        raise ValueError(f"per element diff {absolute_difference} > tolerance {atol}")
