
import numpy as np
from meteor import validate

def test_negentropy_gaussian() -> None:
    n_samples = 100
    samples = np.random.normal(size=n_samples)
    negentropy = validate.negentropy(samples)

    # negentropy should be small for a Gaussian sample
    assert np.abs(negentropy) < 1e-5
