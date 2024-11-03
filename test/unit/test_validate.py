import numpy as np
from numpy.testing import assert_almost_equal

from meteor import validate


def parabolic_objective(x: float) -> float:
    # has a maximum of y = 0 at x = 1
    return -(x**2) + 2 * x - 1


def test_negentropy_gaussian(np_rng: np.random.Generator) -> None:
    n_samples = 10000
    samples = np_rng.normal(size=n_samples)
    negentropy = validate.negentropy(samples)

    # negentropy should be zero for a Gaussian sample
    assert np.abs(negentropy) < 1e-2


def test_negentropy_uniform(np_rng: np.random.Generator) -> None:
    n_samples = 1000000
    samples = np_rng.uniform(size=n_samples)
    negentropy = validate.negentropy(samples)

    uniform_negentropy = (1.0 / 2.0) * np.log(np.pi * np.exp(1) / 6.0)
    assert np.abs(negentropy - uniform_negentropy) < 1e-2


def test_negentropy_zero() -> None:
    negentropy = validate.negentropy(np.zeros(100))
    assert negentropy == -np.inf


def test_negentropy_maximizer_explicit() -> None:
    maximizer = validate.ScalarMaximizer(objective=parabolic_objective)
    test_values = np.linspace(-5, 5, 11)
    maximizer.optimize_over_explicit_values(arguments_to_scan=test_values)
    assert_almost_equal(maximizer.argument_optimum, 1.0)
    assert_almost_equal(maximizer.objective_maximum, 0.0)
    assert list(test_values) == maximizer.values_evaluated
    assert len(maximizer.values_evaluated) == len(maximizer.objective_at_values)


def test_negentropy_maximizer_golden() -> None:
    maximizer = validate.ScalarMaximizer(objective=parabolic_objective)
    maximizer.optimize_with_golden_algorithm(bracket=(-5, 5))
    assert_almost_equal(maximizer.argument_optimum, 1.0, decimal=2)
    assert_almost_equal(maximizer.objective_maximum, 0.0, decimal=2)
    assert len(maximizer.values_evaluated) > 0
    assert len(maximizer.values_evaluated) == len(maximizer.objective_at_values)
