import numpy as np
from scipy.stats import differential_entropy


def negentropy(samples: np.ndarray, tolerance: float = 0.01) -> float:
    """
    Return negentropy (float) of X (numpy array)

    The negetropy is defined as the difference between the entropy of a distribution
    and a Gaussian with same variance.

    citation:
        http://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
    """

    std = np.std(samples.flatten())
    neg_e = (
        0.5 * np.log(2.0 * np.pi * std**2)
        + 0.5
        - differential_entropy(samples.flatten())
    )
    if not neg_e >= -tolerance:
        raise ValueError(
            f"negentropy is a relatively big negative number {neg_e} that exceeds the tolerance {tolerance} -- something may have gone wrong"
        )

    return neg_e
