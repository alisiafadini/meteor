import numpy as np
import reciprocalspaceship as rs

from .scale import scale_datasets
from .validate import ScalarMaximizer, negentropy


def compute_fofo_differences(
    dataset: rs.DataSet,
    *,
    native_amplitudes: str,
    derivative_amplitudes: str,
    calc_amplitudes: str,
    sigf_native: str,
    sigf_deriv: str,
    inplace: bool = False
) -> rs.DataSet | None:
    """
    Compute FoFo difference: DeltaF = derivative_amplitudes_scaled - native_amplitudes_scaled
    with separate uncertainty columns for native and derivative amplitudes.

    Parameters:
    -----------
    dataset : rs.DataSet
        The input dataset containing columns for native amplitudes, derivative amplitudes,
        calculated amplitudes, and uncertainties.
    native_amplitudes : str
        Column label for native amplitudes in the dataset.
    derivative_amplitudes : str
        Column label for derivative amplitudes in the dataset.
    calc_amplitudes : str
        Column label for calculated amplitudes in the dataset.
    sigf_native : str
        Column label for uncertainties of native amplitudes.
    sigf_deriv : str
        Column label for uncertainties of derivative amplitudes.
    """

    # Optionally copy dataset
    if not inplace:
        dataset = dataset.copy()

    # Scale the native amplitudes to the calculated amplitudes
    scaled_native = scale_datasets(
        reference_dataset=dataset,
        dataset_to_scale=dataset,
        column_to_compare=calc_amplitudes,
        uncertainty_column=sigf_native,
    )[native_amplitudes]

    # Scale the derivative amplitudes to the calculated amplitudes
    scaled_derivative = scale_datasets(
        reference_dataset=dataset,
        dataset_to_scale=dataset,
        column_to_compare=calc_amplitudes,
        uncertainty_column=sigf_deriv,
    )[derivative_amplitudes]

    # Compute DeltaF: derivative_amplitudes_scaled - native_amplitudes_scaled
    delta_fofo = scaled_derivative - scaled_native

    # Add DeltaFoFo to the dataset
    dataset["DeltaFoFo"] = delta_fofo

    # Return None if inplace=True, else return modified dataset
    if inplace:
        return None
    else:
        return dataset


def compute_kweights(
    df: rs.DataSeries, sigdf: rs.DataSeries, kweight: float
) -> rs.DataSeries:
    """
    Compute weights for each structure factor based on DeltaF and its uncertainty.
    """
    w = 1 + (sigdf**2 / (sigdf**2).mean()) + kweight * (df**2 / (df**2).mean())
    return w**-1


def compute_kweighted_deltafofo(
    dataset: rs.DataSet,
    *,
    native_amplitudes: str,
    derivative_amplitudes: str,
    calc_amplitudes: str,
    sigf_native: str,
    sigf_deriv: str,
    kweight: float | None = None,
    weight_using_uncertainties: bool = True,
    optimize_kweight: bool = False,
    inplace: bool = False
) -> rs.DataSet | None:
    """
    Compute k-weighted FoFo difference with separate uncertainty columns for native and derivative.

    Parameters:
    -----------
    dataset : rs.DataSet
        The input dataset containing columns for native amplitudes, derivative amplitudes,
        calculated amplitudes, and uncertainties.
    native_amplitudes : str
        Column label for native amplitudes in the dataset.
    derivative_amplitudes : str
        Column label for derivative amplitudes in the dataset.
    calc_amplitudes : str
        Column label for calculated amplitudes in the dataset.
    sigf_native : str
        Column label for uncertainties of native amplitudes.
    sigf_deriv : str
        Column label for uncertainties of derivative amplitudes.
    """

    # Optionally copy dataset
    if not inplace:
        dataset = dataset.copy()

    # Scale the native amplitudes to the calculated amplitudes
    scaled_native = scale_datasets(
        reference_dataset=dataset,
        dataset_to_scale=dataset,
        column_to_compare=calc_amplitudes,
        uncertainty_column=sigf_native,
        weight_using_uncertainties=weight_using_uncertainties,
    )[native_amplitudes]

    # Scale the derivative amplitudes to the calculated amplitudes
    scaled_derivative = scale_datasets(
        reference_dataset=dataset,
        dataset_to_scale=dataset,
        column_to_compare=calc_amplitudes,
        uncertainty_column=sigf_deriv,
        weight_using_uncertainties=weight_using_uncertainties,
    )[derivative_amplitudes]

    # Compute DeltaF: derivative_amplitudes_scaled - native_amplitudes_scaled
    delta_fofo = scaled_derivative - scaled_native

    # Calculate uncertainties for DeltaF
    sigdelta_fofo = np.sqrt(dataset[sigf_deriv] ** 2 + dataset[sigf_native] ** 2)

    # Handle kweight optimization
    if optimize_kweight:

        def negentropy_objective(kweight_value: float) -> float:
            weights = compute_kweights(delta_fofo, sigdelta_fofo, kweight_value)
            weighted_delta_fofo = delta_fofo * weights
            return negentropy(weighted_delta_fofo)

        maximizer = ScalarMaximizer(objective=negentropy_objective)
        maximizer.optimize_with_golden_algorithm(bracket=(0.1, 10.0))
        kweight = maximizer.argument_optimum

    # Compute weights based on DeltaF and uncertainties
    weights = compute_kweights(delta_fofo, sigdelta_fofo, kweight)

    # Apply the weights to DeltaFoFo
    delta_fofo_weighted = delta_fofo * weights

    dataset["DeltaFoFoKWeighted"] = delta_fofo_weighted

    # If inplace is True, return None
    if inplace:
        return None
    else:
        return dataset
