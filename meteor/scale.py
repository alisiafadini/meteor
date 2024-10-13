from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import reciprocalspaceship as rs
import scipy.optimize as opt

if TYPE_CHECKING:
    import pandas as pd

DTYPES_TO_SCALE = (
    rs.AnomalousDifferenceDtype,
    rs.FriedelIntensityDtype,
    rs.FriedelStructureFactorAmplitudeDtype,
    rs.IntensityDtype,
    rs.NormalizedStructureFactorAmplitudeDtype,
    rs.StandardDeviationDtype,
    rs.StandardDeviationFriedelIDtype,
    rs.StandardDeviationFriedelSFDtype,
    rs.StructureFactorAmplitudeDtype,
)
""" automatically scale these types when they appear in an rs.DataSet """


ScaleParameters = Tuple[float, float, float, float, float, float, float]
""" 7x float tuple to hold anisotropic scaling parameters """


def _compute_anisotropic_scale_factors(
    miller_indices: pd.Index, anisotropic_scale_parameters: ScaleParameters
) -> np.ndarray:
    miller_indices_as_array = np.array(list(miller_indices))
    squared_miller_indices = np.square(miller_indices_as_array)

    h_squared = squared_miller_indices[:, 0]
    k_squared = squared_miller_indices[:, 1]
    l_squared = squared_miller_indices[:, 2]

    hk_product = miller_indices_as_array[:, 0] * miller_indices_as_array[:, 1]
    hl_product = miller_indices_as_array[:, 0] * miller_indices_as_array[:, 2]
    kl_product = miller_indices_as_array[:, 1] * miller_indices_as_array[:, 2]

    # Anisotropic scaling term
    exponential_argument = -(
        h_squared * anisotropic_scale_parameters[1]
        + k_squared * anisotropic_scale_parameters[2]
        + l_squared * anisotropic_scale_parameters[3]
        + 2 * hk_product * anisotropic_scale_parameters[4]
        + 2 * hl_product * anisotropic_scale_parameters[5]
        + 2 * kl_product * anisotropic_scale_parameters[6]
    )

    return anisotropic_scale_parameters[0] * np.exp(exponential_argument)


def compute_scale_factors(
    *,
    reference_values: rs.DataSeries,
    values_to_scale: rs.DataSeries,
    reference_uncertainties: rs.DataSeries | None = None,
    to_scale_uncertainties: rs.DataSeries | None = None,
) -> rs.DataSeries:
    """
    Compute anisotropic scale factors to modify `values_to_scale` to be on the same scale as
    `reference_values`.

    Following SCALEIT, the scaling model is an anisotropic model, applying a transformation of the
    form:

        C * exp{ -(h**2 B11 + k**2 B22 + l**2 B33 +
                    2hk B12 + 2hl  B13 +  2kl B23) }

    The parameters Bxy are fit using least squares, optionally with uncertainty weighting.

    Parameters
    ----------
    reference_values : rs.DataSeries
        The reference dataset against which scaling is performed, indexed by Miller indices.
    values_to_scale : rs.DataSeries
        The dataset to be scaled, also Miller indexed.
    reference_uncertainties : rs.DataSeries, optional
        Uncertainty values associated with `reference_values`. If provided, they are used in
        weighting the scaling process. Must have the same index as `reference_values`.
    to_scale_uncertainties : rs.DataSeries, optional
        Uncertainty values associated with `values_to_scale`. If provided, they are used in
        weighting the scaling process. Must have the same index as `values_to_scale`.

    Returns
    -------
    rs.DataSeries
        The computed anisotropic scale factors for each Miller index in `values_to_scale`.

    See Also
    --------
    scale_datasets : higher-level interface that operates on entire DataSets, typically more
    convienent.

    Citations:
    ----------
    [1] SCALEIT https://www.ccp4.ac.uk/html/scaleit.html
    """
    common_miller_indices = reference_values.index.intersection(values_to_scale.index)

    # if we are going to weight the scaling using the uncertainty values, then the weights will be
    #    inverse_sigma = 1 / sqrt{ sigmaA ** 2 + sigmaB ** 2 }
    if reference_uncertainties is not None and to_scale_uncertainties is not None:

        if not reference_uncertainties.index.equals(reference_values.index):
            msg = "indices of `reference_uncertainties`, `reference_values` differ, cannot combine"
            raise IndexError(msg)
        if not to_scale_uncertainties.index.equals(values_to_scale.index):
            msg = "indices of `to_scale_uncertainties`, `values_to_scale` differ, cannot combine"
            raise IndexError(msg)

        uncertainty_weights = np.sqrt(
            np.square(reference_uncertainties.loc[common_miller_indices])
            + np.square(to_scale_uncertainties.loc[common_miller_indices])
        )

    else:
        uncertainty_weights = 1.0

    common_reference_values = reference_values.loc[common_miller_indices]
    common_values_to_scale = values_to_scale.loc[common_miller_indices]

    def compute_residuals(scaling_parameters: ScaleParameters) -> np.ndarray:
        scale_factors = _compute_anisotropic_scale_factors(
            common_miller_indices, scaling_parameters
        )
        return uncertainty_weights * (
            scale_factors * common_values_to_scale - common_reference_values
        )

    initial_scaling_parameters: ScaleParameters = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    optimization_result = opt.least_squares(compute_residuals, initial_scaling_parameters)
    optimized_parameters: ScaleParameters = optimization_result.x

    # now be sure to compute the scale factors for all miller indices in `values_to_scale`
    optimized_scale_factors = _compute_anisotropic_scale_factors(
        values_to_scale.index, optimized_parameters
    )

    if len(optimized_scale_factors) != len(values_to_scale):
        msg1 = "length mismatch: `optimized_scale_factors`"
        msg2 = f"({len(optimized_scale_factors)}) vs `values_to_scale` ({len(values_to_scale)})"
        raise RuntimeError(msg1, msg2)

    return optimized_scale_factors


def scale_datasets(
    *,
    reference_dataset: rs.DataSet,
    dataset_to_scale: rs.DataSet,
    column_to_compare: str = "F",
    uncertainty_column: str = "SIGF",
    weight_using_uncertainties: bool = True,
) -> rs.DataSet:
    """
    Scale a dataset to align it with a reference dataset using anisotropic scaling.

    This function scales the dataset (`dataset_to_scale`) by comparing it to a reference dataset
    (`reference_dataset`) based on a specified column. The scaling applies an anisotropic model of
    the form:

        C * exp{ -(h**2 B11 + k**2 B22 + l**2 B33 +
                    2hk B12 + 2hl  B13 +  2kl B23) }

    The parameters Bxy are fit using least squares, optionally with uncertainty weighting.

    NB! All intensity, amplitude, and standard deviation columns in `dataset_to_scale` will be
    modified (scaled). To access the scale parameters directly, use
    `meteor.scale.compute_scale_factors`.

    Parameters
    ----------
    reference_dataset : rs.DataSet
        The reference dataset, containing a column with the values to which `dataset_to_scale` is
        compared.
    dataset_to_scale : rs.DataSet
        The dataset to be scaled. It must contain the same columns as `reference_dataset`.
    column_to_compare : str, optional (default: "F")
        The column in both datasets that is used for the comparison and scaling.
    uncertainty_column : str, optional (default: "SIGF")
        The column in both datasets containing uncertainty values for weighting. Used only if
        `weight_using_uncertainties` is True.
    weight_using_uncertainties : bool, optional (default: True)
        Whether or not to weight the scaling by uncertainty values. If True, uncertainty values are
        extracted from the `uncertainty_column` in both datasets.

    Returns
    -------
    rs.DataSet
        A copy of `dataset_to_scale`, with the specified columns scaled to match the reference
        dataset.

    See Also
    --------
    compute_scale_factors : function to compute scale factors directly

    Citations:
    ----------
    [1] SCALEIT https://www.ccp4.ac.uk/html/scaleit.html
    """
    if weight_using_uncertainties:
        scale_factors = compute_scale_factors(
            reference_values=reference_dataset[column_to_compare],
            values_to_scale=dataset_to_scale[column_to_compare],
            reference_uncertainties=reference_dataset[uncertainty_column],
            to_scale_uncertainties=dataset_to_scale[uncertainty_column],
        )
    else:
        scale_factors = compute_scale_factors(
            reference_values=reference_dataset[column_to_compare],
            values_to_scale=dataset_to_scale[column_to_compare],
            reference_uncertainties=None,
            to_scale_uncertainties=None,
        )

    scaled_dataset = dataset_to_scale.copy()
    columns_to_scale = [
        column_name
        for column_name, column_dtype in dataset_to_scale.dtypes.items()
        if isinstance(column_dtype, DTYPES_TO_SCALE)
    ]
    if column_to_compare not in columns_to_scale:
        msg = (
            f"the `column_to_compare` {column_to_compare} not flagged by dtype as a column to scale"
        )
        raise TypeError(msg)

    for column in columns_to_scale:
        scaled_dataset[column] *= scale_factors

    return scaled_dataset
