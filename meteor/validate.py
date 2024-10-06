from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import differential_entropy


def negentropy(samples: np.ndarray, tolerance: float = 0.1) -> float:
    """Computes the negentropy of a given sample array.

    Negentropy is defined as the difference between the entropy of a given
    distribution and the entropy of a Gaussian distribution with the same variance.
    It is a measure of non-Gaussianity, with higher values indicating greater deviation
    from Gaussianity.

    Args:
    ----
        samples (np.ndarray): A numpy array of sample data for which to calculate the negentropy.
        tolerance (float): Tolerance level determining if the negentropy is suspiciously negative.
                           Defaults to 0.01.

    Returns:
    -------
        float: The computed negentropy of the sample data.

    Raises:
    ------
        ValueError: If the computed negentropy is less than the negative tolerance,
                    indicating potential issues with the computation.

    References:
    ----------
        http://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/

    Example:
    -------
        >>> samples = np.random.normal(size=1000)
        >>> negentropy(samples)
        0.0012  # Example output, varies with input samples.

    """
    std = np.std(samples.flatten())
    if std <= 0.0:
        return -np.inf

    neg_e = 0.5 * np.log(2.0 * np.pi * std**2) + 0.5 - differential_entropy(samples.flatten())
    if not neg_e >= -tolerance:
        raise ValueError(
            f"negentropy is a large negative number {neg_e}, exceeds the tolerance {tolerance}"
            " -- something may have gone wrong"
        )

    return neg_e


class ScalarMaximizer:
    """
    Maximize a function using one of two strategies, or a combination of both:

      1. simply loop over an explicit list of argument values and pick the max
      2. use the Golden ratio method to try and be a bit faster

    The objective function to maximize must be a scalar mapping, taking a single float as an
    argument and returning a single float. In the context of METEOR, this is useful to e.g.
    maximize over simple parameter choices, such as TV regularization strength or k-weight values.

    Attributes:
    ----------
    objective : Callable[[float], float]
        The objective function to maximize.
    argument_optimum : float
        The argument (input) that gives the highest objective value found so far.
    objective_maximum : float
        The maximum value of the objective function found so far.
    values_evaluated : set[float]
        A set of argument values that have been evaluated during optimization.
    """

    def __init__(self, *, objective: Callable[[float], float]):
        """
        Initializes the Maximizer with the given objective function.

        Parameters:
        ----------
        objective : Callable[[float], float]
            The objective function to be maximized. It should map a single float to a float.
        """
        self.objective = objective
        self.argument_optimum: float = np.nan
        self.objective_maximum: float = -np.inf
        self.values_evaluated: set[float] = set()

    def _update_optima(self, argument_test_value):
        objective_value = self.objective(argument_test_value)
        if objective_value > self.objective_maximum:
            self.argument_optimum = argument_test_value
            self.objective_maximum = objective_value

    def optimize_over_explicit_values(self, *, arguments_to_scan: Sequence[float] | np.ndarray):
        """
        Scans through a list or array of argument values to find the optimum.

        Parameters:
        ----------
        arguments_to_scan : list[float] | np.ndarray
            A list or array of argument values to evaluate.
        """
        for argument_test_value in arguments_to_scan:
            self._update_optima(argument_test_value)
            self.values_evaluated.add(argument_test_value)

    def optimize_with_golden_algorithm(self, *, bracket: tuple[float, float]):
        """
        Uses the golden-section search algorithm to maximize the objective function within a given
        bracket.

        Parameters:
        ----------
        bracket : tuple[float, float]
            A tuple containing the lower and upper bounds of the bracket within which to search
            for the optimum.

        Raises:
        -------
        RuntimeError
            If the optimization using the golden algorithm fails.
        """

        def _objective_with_value_tracking(argument_test_value):
            """adds the evaluated value to self.values_evaluated"""
            self.values_evaluated.add(argument_test_value)
            return -self.objective(argument_test_value)  # negative: we want max

        optimizer_result = minimize_scalar(
            _objective_with_value_tracking, bracket=bracket, method="golden"
        )
        if not optimizer_result.success:
            raise RuntimeError("Golden minimization failed")

        self._update_optima(optimizer_result.x)
