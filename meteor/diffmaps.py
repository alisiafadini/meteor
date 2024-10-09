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
    uncertainty_column: str = "SIGF",
    weight_using_uncertainties: bool = True,
    inplace: bool = False
) -> rs.DataSet | None:
    """
    Compute FoFo difference:
    DeltaF = derivative_amplitudes_scaled - native_amplitudes_scaled
    and return a new dataset with the DeltaFoFo column,
    or modify the dataset in place if `inplace` is True.

    Parameters:
    -----------
    dataset : rs.DataSet
        The input dataset containing columns for native amplitudes, derivative amplitudes,
        calculated amplitudes, and calculated phases.
    native_amplitudes : str
        The column label for native amplitudes in the dataset.
    derivative_amplitudes : str
        The column label for derivative amplitudes in the dataset.
    calc_amplitudes : str
        The column label for calculated amplitudes in the dataset.
    uncertainty_column : str, optional (default: "SIGF")
        The column containing uncertainty values. Used for scaling.
    weight_using_uncertainties : bool, optional (default: True)
        Whether or not to weight the scaling by uncertainty values.
    inplace : bool, optional (default: False)
        Whether to modify the dataset in place or return a new copy.

    Returns:
    --------
    rs.DataSet | None
        Returns a new dataset with an additional DeltaFoFo column if `inplace` is False, or modifies
        the original dataset in place if `inplace` is True.
    """

    # Optionally copy dataset
    if not inplace:
        dataset = dataset.copy()

    # Scale the native amplitudes to the calculated amplitudes
    scaled_native = scale_datasets(
        reference_dataset=dataset,
        dataset_to_scale=dataset,
        column_to_compare=calc_amplitudes,
        uncertainty_column=uncertainty_column,
        weight_using_uncertainties=weight_using_uncertainties,
    )[native_amplitudes]

    # Scale the derivative amplitudes to the calculated amplitudes
    scaled_derivative = scale_datasets(
        reference_dataset=dataset,
        dataset_to_scale=dataset,
        column_to_compare=calc_amplitudes,
        uncertainty_column=uncertainty_column,
        weight_using_uncertainties=weight_using_uncertainties,
    )[derivative_amplitudes]

    # Compute DeltaF: derivative_amplitudes_scaled - native_amplitudes_scaled
    delta_fofo = scaled_derivative - scaled_native

    # Add DeltaF to the dataset
    dataset["DeltaFoFo"] = delta_fofo

    # If inplace, return None (as dataset is modified in place), otherwise return dataset
    if not inplace:
        return dataset
    else:
        return None


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
    uncertainty_column: str = "SIGF",
    kweight: float | None = None,
    weight_using_uncertainties: bool = True,
    optimize_kweight: bool = False,
    inplace: bool = False
) -> rs.DataSet | None:
    """
    Compute k-weighted FoFo difference:
    DeltaF = derivative_amplitudes_scaled - native_amplitudes_scaled
    Apply weighting based on uncertainties and optionally optimize kweight.

    Parameters:
    -----------
    dataset : rs.DataSet
        The input dataset containing columns for native amplitudes, derivative amplitudes,
        calculated amplitudes, and uncertainties.
    native_amplitudes : str
        The column label for native amplitudes in the dataset.
    derivative_amplitudes : str
        The column label for derivative amplitudes in the dataset.
    calc_amplitudes : str
        The column label for calculated amplitudes in the dataset.
    uncertainty_column : str, optional (default: "SIGF")
        The column containing uncertainty values. Used for scaling.
    kweight : float, optional
        A fixed value of kweight to apply. If None, it will be optimized if `optimize_kweight` is True.
    weight_using_uncertainties : bool, optional (default: True)
        Whether or not to weight the scaling by uncertainty values.
    optimize_kweight : bool, optional (default: False)
        Whether to optimize the kweight value to maximize negentropy.
    inplace : bool, optional (default: False)
        Whether to modify the dataset in place or return a new copy.

    Returns:
    --------
    rs.DataSet | None
        Returns a new dataset with an additional DeltaFoFoKWeighted column if `inplace` is False,
        or modifies the original dataset in place if `inplace` is True.
    """

    # Optionally copy dataset
    if not inplace:
        dataset = dataset.copy()

    # Scale the native amplitudes to the calculated amplitudes
    scaled_native = scale_datasets(
        reference_dataset=dataset,
        dataset_to_scale=dataset,
        column_to_compare=calc_amplitudes,
        uncertainty_column=uncertainty_column,
        weight_using_uncertainties=weight_using_uncertainties,
    )[native_amplitudes]

    # Scale the derivative amplitudes to the calculated amplitudes
    scaled_derivative = scale_datasets(
        reference_dataset=dataset,
        dataset_to_scale=dataset,
        column_to_compare=calc_amplitudes,
        uncertainty_column=uncertainty_column,
        weight_using_uncertainties=weight_using_uncertainties,
    )[derivative_amplitudes]

    # Compute DeltaF: derivative_amplitudes_scaled - native_amplitudes_scaled
    delta_fofo = scaled_derivative - scaled_native

    # Get uncertainties for DeltaF
    sigdelta_fofo = np.sqrt(
        dataset[uncertainty_column][derivative_amplitudes] ** 2
        + dataset[uncertainty_column][native_amplitudes] ** 2
    )

    # Define negentropy objective for kweight optimization
    def negentropy_objective(kweight_value: float) -> float:
        weights = compute_kweights(delta_fofo, sigdelta_fofo, kweight_value)
        weighted_delta_fofo = delta_fofo * weights
        return negentropy(weighted_delta_fofo)

    # Optimize kweight if requested
    if optimize_kweight:
        maximizer = ScalarMaximizer(objective=negentropy_objective)
        # For this example, we use a reasonable bracket for kweight; it may need adjustment
        maximizer.optimize_with_golden_algorithm(bracket=(0.1, 10.0))
        kweight = maximizer.argument_optimum

    # Compute kweights with the given or optimized kweight
    if kweight is None:
        raise ValueError("kweight must be provided or optimized")

    weights = compute_kweights(delta_fofo, sigdelta_fofo, kweight)

    # Compute weighted DeltaFoFo
    delta_fofo_weighted = delta_fofo * weights

    # Add weighted DeltaF to the dataset
    dataset["DeltaFoFoKWeighted"] = delta_fofo_weighted

    # If inplace, return None (as dataset is modified in place), otherwise return dataset
    if not inplace:
        return dataset
    else:
        return None
