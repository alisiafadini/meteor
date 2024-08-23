import numpy as np

from meteor import validate


def test_negentropy_gaussian() -> None:
    n_samples = 10000
    samples = np.random.normal(size=n_samples)
    negentropy = validate.negentropy(samples)

    # negentropy should be zero for a Gaussian sample
    assert np.abs(negentropy) < 1e-2


def test_negentropy_uniform() -> None:
    n_samples = 1000000
    samples = np.random.uniform(size=n_samples)
    negentropy = validate.negentropy(samples)

    uniform_negentropy = (1.0 / 2.0) * np.log(np.pi * np.exp(1) / 6.0)
    assert np.abs(negentropy - uniform_negentropy) < 1e-2


def test_negentropy_zero() -> None:
    negentropy = validate.negentropy(np.zeros(100))
    assert negentropy == -np.inf
