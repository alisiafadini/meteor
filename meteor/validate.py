import numpy as np
from scipy.stats import differential_entropy


def negentropy(samples: np.ndarray, tolerance: float = 0.01) -> float:
    """
    Computes the negentropy of a given sample array.

    Negentropy is defined as the difference between the entropy of a given
    distribution and the entropy of a Gaussian distribution with the same variance.
    It is a measure of non-Gaussianity, with higher values indicating greater deviation
    from Gaussianity.

    Args:
        samples (np.ndarray): A numpy array of sample data for which to calculate the negentropy.
        tolerance (float): A tolerance level for checking if the negentropy is suspiciously negative.
                           Defaults to 0.01.

    Returns:
        float: The computed negentropy of the sample data.

    Raises:
        ValueError: If the computed negentropy is less than the negative tolerance,
                    indicating potential issues with the computation.

    References:
        http://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/

    Example:
        >>> samples = np.random.normal(size=1000)
        >>> negentropy(samples)
        0.0012  # Example output, varies with input samples.
    """

    std = np.std(samples.flatten())
    if std <= 0.0:
        return np.inf
    
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
