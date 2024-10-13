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
    native_phases_column: str,
    native_uncertainty_column: str | None = None,
    derivative_amplitudes_column: str,
    derivative_phases_column: str | None = None,
    derivative_uncertainty_column: str | None = None,
    output_amplitudes_column: str = "DF",
    output_phases_column: str = "DPHI",
    output_uncertainty_column: str = "SIGDF",
) -> rs.DataSet:
    """
    Computes amplitude and phase differences between native and derivative structure factor sets.

    If `derivative_phases_column` are not supplied....
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

    # compute complex differences & convert back to amplitude and phase DataSeries
    delta_complex = derivative_complex - native_complex
    delta_amplitudes, delta_phases = complex_array_to_rs_dataseries(
        delta_complex, dataset.index
    )

    dataset[output_amplitudes_column] = delta_amplitudes
    dataset[output_phases_column] = delta_phases

    if (derivative_uncertainty_column is not None) and (native_uncertainty_column is not None):
        sigdelta_amplitudes = np.sqrt(
            dataset[derivative_uncertainty_column] ** 2 + dataset[native_uncertainty_column] ** 2
        )
        dataset[output_uncertainty_column] = sigdelta_amplitudes

    return dataset


def compute_kweights(
    delta_amplitudes: rs.DataSeries, delta_uncertainties: rs.DataSeries, k_parameter: float
) -> rs.DataSeries:
    """
    Compute weights for each structure factor based on DeltaF and its uncertainty.

    Parameters
    ----------
    delta_amplitudes: rs.DataSeries
        The series representing the structure factor differences (DeltaF).

    sigdelta_amplitudes: rs.DataSeries
        Representing the uncertainties (sigma) of the structure factor differences.

    k_parameter: float
        A scaling factor applied to the squared `df` values in the weight calculation.

    Returns
    -------
    rs.DataSeries:
        A series of computed weights, where higher uncertainties and larger differences lead to 
        lower weights.
    """
    w = (
        1
        + (delta_uncertainties**2 / (delta_uncertainties**2).mean())
        + k_parameter * (delta_amplitudes**2 / (delta_amplitudes**2).mean())
    )
    return 1.0 / w


def compute_kweighted_difference_map(
    dataset: rs.DataSet,
    *,
    k_parameter: float,
    native_amplitudes_column: str,
    native_phases_column: str,
    native_uncertainty_column: str,
    derivative_amplitudes_column: str,
    derivative_phases_column: str | None = None,
    derivative_uncertainty_column: str,
    output_amplitudes_column: str = "DF_KWeighted",
    output_phases_column: str = "DPHI_KWeighted",
    output_uncertainties_column: str = "SIGDF_KWeighted",
) -> rs.DataSet:
    """
    Compute k-weighted differences between native and derivative amplitudes and phases.

    Assumes that scaling has already been applied to the amplitudes before calling this function.

    Need to either specify k-weight or enable optimization. Dataset modified inplace.

    Parameters
    ----------
    dataset : rs.DataSet
        The input dataset containing columns for native and derivative amplitudes/phases.
    k_parameter : float
        ...
    native_amplitudes_column : str
        Column label for native amplitudes in the dataset.
    derivative_amplitudes_column : str
        Column label for derivative amplitudes in the dataset.
    native_phases_column : str, optional
        Column label for native phases.
    derivative_phases_column : str, optional
        Column label for derivative phases, by default None.
    uncertainty_native_column : str
        Column label for uncertainties of native amplitudes.
    uncertainty_deriv_column : str
        Column label for uncertainties of derivative amplitudes.

    Returns
    -------
    dataset: rs.DataSet
        dataset with added columns
    """
    
    # this label is only used internally in this function
    # less than ideal...
    diffmap_amplitudes = "__INTERNAL_DF_LABEL"

    diffmap_dataset = compute_difference_map(
        dataset=dataset,
        native_amplitudes_column=native_amplitudes_column,
        native_phases_column=native_phases_column,
        native_uncertainty_column=native_uncertainty_column,
        derivative_amplitudes_column=derivative_amplitudes_column,
        derivative_phases_column=derivative_phases_column,
        derivative_uncertainty_column=derivative_uncertainty_column,
        output_amplitudes_column=diffmap_amplitudes,
        output_phases_column=output_phases_column,
        output_uncertainty_column=output_uncertainties_column,
    )

    weights = compute_kweights(
        diffmap_dataset[diffmap_amplitudes],
        diffmap_dataset[output_uncertainties_column],
        k_parameter
    )

    output_ds = dataset.copy()
    output_ds[output_amplitudes_column] = diffmap_dataset[diffmap_amplitudes] * weights
    output_ds[output_phases_column] = diffmap_dataset[output_phases_column]
    output_ds[output_uncertainties_column] = diffmap_dataset[output_uncertainties_column]

    return output_ds


# def max_negentropy_kweighted_difference_map(
#     dataset: rs.DataSet,
#     *,
#     native_amplitudes_column: str,
#     native_phases_column: str,
#     native_uncertainty_column: str,
#     derivative_amplitudes_column: str,
#     derivative_phases_column: str | None = None,
#     derivative_uncertainty_column: str,
#     output_amplitudes_column: str = "DF_KWeighted",
#     output_phases_column: str = "DPHI_KWeighted",
#     output_uncertainties_column: str = "SIGDF_KWeighted",
# ) -> rs.DataSet:
#     """
#     Compute k-weighted differences between native and derivative amplitudes and phases.

#     Determines an "optimal" k_parameter, between 0.0 and 1.0, that maximizes the resulting
#     difference map negentropy. Assumes that scaling has already been applied to the amplitudes
#     before calling this function.

#     Parameters
#     ----------
#     dataset : rs.DataSet
#         The input dataset containing columns for native and derivative amplitudes/phases.
#     native_amplitudes_column : str
#         Column label for native amplitudes in the dataset.
#     derivative_amplitudes_column : str
#         Column label for derivative amplitudes in the dataset.
#     native_phases_column : str, optional
#         Column label for native phases.
#     derivative_phases_column : str, optional
#         Column label for derivative phases, by default None.
#     uncertainty_native_column : str
#         Column label for uncertainties of native amplitudes.
#     uncertainty_deriv_column : str
#         Column label for uncertainties of derivative amplitudes.
#     k_parameter : float, optional
#         k-weight factor, optional.

#     Returns
#     -------
#     kweighted_dataset: rs.DataSet
#         dataset with added columns

#     opt_k_parameter: float
#         optimized weight
#     """

#     def negentropy_objective(k_parameter: float) -> float:
#             # Apply k-weighting to differences
#             weights = compute_kweights(
#                 delta_amplitudes, sigdelta_amplitudes, k_parameter
#             )
#             dataset[output_weighted_amplitudes_column] = delta_amplitudes * weights

#             # Convert weighted amplitudes and phases to a map
#             delta_map = compute_map_from_coefficients(
#                 map_coefficients=dataset,
#                 amplitude_label=output_weighted_amplitudes_column,
#                 phase_label=native_phases_column,
#                 map_sampling=TV_MAP_SAMPLING,
#             )

#             delta_map_as_array = np.array(delta_map.grid)

#             # Compute negentropy of the map
#             return negentropy(delta_map_as_array.flatten())

#     # Optimize k_parameter using negentropy objective
#     maximizer = ScalarMaximizer(objective=negentropy_objective)
#     maximizer.optimize_over_explicit_values(
#         arguments_to_scan=np.linspace(0.0, 1.0, 101)
#     )

#     print(maximizer.argument_optimum, maximizer.objective_maximum, maximizer.values_evaluated)

#     kweighted_dataset = compute_kweighted_difference_map(
#         dataset=dataset,
#         k_parameter=opt_k_parameter,
#         native_amplitudes_column=native_amplitudes_column,
#         derivative_amplitudes_column=derivative_amplitudes_column,
#         native_phases_column=native_phases_column,
#         derivative_phases_column=derivative_phases_column,
#         uncertainty_native_column=uncertainty_native_column,
#         uncertainty_deriv_column=uncertainty_deriv_column,
#         output_unweighted_amplitudes_column=output_unweighted_amplitudes_column,
#         output_weighted_amplitudes_column=output_weighted_amplitudes_column,
#     )

#     return kweighted_dataset, opt_k_parameter