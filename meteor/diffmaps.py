from typing import Literal

import numpy as np
import pandas as pd
import reciprocalspaceship as rs

from .settings import TV_MAP_SAMPLING
from .utils import (
    complex_array_to_rs_dataseries,
    compute_map_from_coefficients,
    rs_dataseries_to_complex_array,
)
from .validate import ScalarMaximizer, negentropy


def compute_difference_map(
    dataset: rs.DataSet,
    *,
    native_amplitudes_column: str,
    derivative_amplitudes_column: str,
    native_phases_column: str,
    derivative_phases_column: str | None = None,
    output_amplitudes_column: str = "DF",
    output_phases_column: str = "DPHI",
) -> rs.DataSet:
    """
    Computes amplitude and phase differences between native and derivative structure factor sets.

    This function calculates the complex differences between native and derivative
    datasets using their respective amplitude and phase columns. The results are either
    modified in-place.

    Args:
        dataset (rs.DataSet): The input dataset containing native and derivative amplitudes
            and phases.
        native_amplitudes_column (str): The name of the column containing native amplitudes.
        derivative_amplitudes_column (str): The name of the column containing derivative amplitudes.
        native_phases_column (str): The name of the column containing native phases.
        derivative_phases_column (str | None, optional): The name of the column containing
            derivative phases. If `None`, the native phases are used for the derivative.
            Defaults to None.
        output_amplitudes_column (str, optional): The name of the output column where
            amplitude differences (DF) will be stored. Defaults to "DF".
        output_phases_column (str, optional): The name of the output column where phase
            differences (DPHI) will be stored. Defaults to "DPHI".
    Return:
        dataset with added columns
    """

    dataset = dataset.copy()

    # Convert native and derivative amplitude/phase pairs to complex arrays
    native_complex = rs_dataseries_to_complex_array(
        dataset[native_amplitudes_column], dataset[native_phases_column]
    )

    if derivative_phases_column is not None:
        derivative_complex = rs_dataseries_to_complex_array(
            dataset[derivative_amplitudes_column], dataset[derivative_phases_column]
        )
    else:
        # If no derivative phases are provided, assume they are the same as native phases
        derivative_complex = rs_dataseries_to_complex_array(
            dataset[derivative_amplitudes_column], dataset[native_phases_column]
        )

    # Compute complex differences
    delta_complex = derivative_complex - native_complex

    # Convert back to amplitude and phase DataSeries
    delta_amplitudes, delta_phases = complex_array_to_rs_dataseries(
        delta_complex, dataset.index
    )

    # Add results to dataset
    dataset[output_amplitudes_column] = delta_amplitudes
    dataset[output_phases_column] = delta_phases

    return dataset


def compute_kweights(
    deltaf: rs.DataSeries, sigdeltaf: rs.DataSeries, kweight: float
) -> rs.DataSeries:
    """
    Compute weights for each structure factor based on DeltaF and its uncertainty.

    Args:
        deltaf (rs.DataSeries): The series representing the structure factor differences (DeltaF).
        sigdeltaf (rs.DataSeries):Representing the uncertainties (sigma)
                                of the structure factor differences.
        kweight (float): A scaling factor applied to the squared `df` values
                        in the weight calculation.

    Returns:
        rs.DataSeries: A series of computed weights,
                       where higher uncertainties and larger differences
                       lead to lower weights.
    """
    w = (
        1
        + (sigdeltaf**2 / (sigdeltaf**2).mean())
        + kweight * (deltaf**2 / (deltaf**2).mean())
    )
    return w**-1


def compute_kweighted_difference_map(
    dataset: rs.DataSet,
    *,
    kweight: float,
    native_amplitudes_column: str,
    derivative_amplitudes_column: str,
    native_phases_column: str,
    derivative_phases_column: str | None = None,
    sigf_native_column: str,
    sigf_deriv_column: str,
    output_unweighted_amplitudes_column: str = "DF",
    output_weighted_amplitudes_column: str = "DFKWeighted",
) -> rs.DataSet:
    """
    Compute k-weighted differences between native and derivative amplitudes and phases.

    Assumes that scaling has already been applied to the amplitudes before calling this function.

    Need to either specify k-weight or enable optimization. Dataset modified inplace.

    Parameters
    ----------
    dataset : rs.DataSet
        The input dataset containing columns for native and derivative amplitudes/phases.
    native_amplitudes_column : str
        Column label for native amplitudes in the dataset.
    derivative_amplitudes_column : str
        Column label for derivative amplitudes in the dataset.
    native_phases_column : str, optional
        Column label for native phases.
    derivative_phases_column : str, optional
        Column label for derivative phases, by default None.
    sigf_native_column : str
        Column label for uncertainties of native amplitudes.
    sigf_deriv_column : str
        Column label for uncertainties of derivative amplitudes.
    kweight : float, optional
        k-weight factor, optional.

    Returns
    -------
    dataset: rs.DataSet
        dataset with added columns
    """
    dataset = dataset.copy()

    # Compute differences between native and derivative amplitudes and phases
    dataset = compute_difference_map(
        dataset=dataset,
        native_amplitudes_column=native_amplitudes_column,
        derivative_amplitudes_column=derivative_amplitudes_column,
        native_phases_column=native_phases_column,
        derivative_phases_column=derivative_phases_column,
        output_amplitudes_column=output_unweighted_amplitudes_column,
    )

    delta_amplitudes = dataset[output_unweighted_amplitudes_column]
    sigdelta_amplitudes = np.sqrt(
        dataset[sigf_deriv_column] ** 2 + dataset[sigf_native_column] ** 2
    )

    # Compute weights and apply to differences
    weights = compute_kweights(delta_amplitudes, sigdelta_amplitudes, kweight)
    dataset[output_weighted_amplitudes_column] = delta_amplitudes * weights

    return dataset


def _determine_kweight(
    dataset: rs.DataSet,
    delta_amplitudes: str,
    sigdelta_amplitudes: str,
    output_weighted_amplitudes_column: str,
    native_phases_column: str,
) -> float:

    def negentropy_objective(kweight_value: float) -> float:
        # Apply k-weighting to differences
        weights = compute_kweights(
            delta_amplitudes, sigdelta_amplitudes, kweight_value
        )
        dataset[output_weighted_amplitudes_column] = delta_amplitudes * weights

        # Convert weighted amplitudes and phases to a map
        delta_map = compute_map_from_coefficients(
            map_coefficients=dataset,
            amplitude_label=output_weighted_amplitudes_column,
            phase_label=native_phases_column,
            map_sampling=TV_MAP_SAMPLING,
        )

        delta_map_as_array = np.array(delta_map.grid)

        # Compute negentropy of the map
        return negentropy(delta_map_as_array.flatten())

    # Optimize kweight using negentropy objective
    maximizer = ScalarMaximizer(objective=negentropy_objective)
    maximizer.optimize_over_explicit_values(
        arguments_to_scan=np.linspace(0.0, 1.0, 101)
    )
    print(maximizer.argument_optimum, maximizer.objective_maximum, maximizer.values_evaluated)

    return maximizer.argument_optimum


def max_negentropy_kweighted_difference_map(
    dataset: rs.DataSet,
    *,
    native_amplitudes_column: str,
    derivative_amplitudes_column: str,
    native_phases_column: str,
    derivative_phases_column: str | None = None,
    sigf_native_column: str,
    sigf_deriv_column: str,
    output_unweighted_amplitudes_column: str = "DF",
    output_weighted_amplitudes_column: str = "DFKWeighted",
) -> rs.DataSet:
    """
    Compute k-weighted differences between native and derivative amplitudes and phases.

    Assumes that scaling has already been applied to the amplitudes before calling this function.

    Need to either specify k-weight or enable optimization. Dataset modified inplace.

    Parameters
    ----------
    dataset : rs.DataSet
        The input dataset containing columns for native and derivative amplitudes/phases.
    native_amplitudes_column : str
        Column label for native amplitudes in the dataset.
    derivative_amplitudes_column : str
        Column label for derivative amplitudes in the dataset.
    native_phases_column : str, optional
        Column label for native phases.
    derivative_phases_column : str, optional
        Column label for derivative phases, by default None.
    sigf_native_column : str
        Column label for uncertainties of native amplitudes.
    sigf_deriv_column : str
        Column label for uncertainties of derivative amplitudes.
    kweight : float, optional
        k-weight factor, optional.

    Returns
    -------
    kweighted_dataset: rs.DataSet
        dataset with added columns

    kweight: float
        optimized weight
    """

    dataset = compute_difference_map(
        dataset=dataset,
        native_amplitudes_column=native_amplitudes_column,
        derivative_amplitudes_column=derivative_amplitudes_column,
        native_phases_column=native_phases_column,
        derivative_phases_column=derivative_phases_column,
        output_amplitudes_column=output_unweighted_amplitudes_column,
    )

    delta_amplitudes = dataset[output_unweighted_amplitudes_column]
    sigdelta_amplitudes = np.sqrt(
        dataset[sigf_deriv_column] ** 2 + dataset[sigf_native_column] ** 2
    )

    kweight = _determine_kweight(
        dataset,
        delta_amplitudes,
        sigdelta_amplitudes,
        output_weighted_amplitudes_column,
        native_phases_column,
    )

    kweighted_dataset = compute_kweighted_difference_map(
        dataset=dataset,
        kweight=kweight,
        native_amplitudes_column=native_amplitudes_column,
        derivative_amplitudes_column=derivative_amplitudes_column,
        native_phases_column=native_phases_column,
        derivative_phases_column=derivative_phases_column,
        sigf_native_column=sigf_native_column,
        sigf_deriv_column=sigf_deriv_column,
        output_unweighted_amplitudes_column=output_unweighted_amplitudes_column,
        output_weighted_amplitudes_column=output_weighted_amplitudes_column,
    )

    return kweighted_dataset, kweight