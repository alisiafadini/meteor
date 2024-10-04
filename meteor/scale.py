from typing import Final, Tuple

import numpy as np
import pandas as pd
import reciprocalspaceship as rs
import scipy.optimize as opt

DTYPES_TO_SCALE: Final[list[rs.MTZRealDtype]] = [
    rs.AnomalousDifferenceDtype(),
    rs.FriedelIntensityDtype(),
    rs.FriedelStructureFactorAmplitudeDtype(),
    rs.IntensityDtype(),
    rs.NormalizedStructureFactorAmplitudeDtype(),
    rs.StandardDeviationDtype(),
    rs.StandardDeviationFriedelIDtype(),
    rs.StandardDeviationFriedelSFDtype(),
    rs.StructureFactorAmplitudeDtype(),
]
""" automatically scale these types when they appear in an rs.DataSet """


ScaleParameters = Tuple[float, float, float, float, float, float, float]
""" 7x float tuple to hold anisotropic scaling parameters """


def _compute_anisotropic_scale_factors(
    miller_indices: pd.Index, anisotropic_scale_parameters: ScaleParameters
) -> np.ndarray:
    for miller_index in ["H", "K", "L"]:
        assert miller_index in miller_indices

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
    common_miller_indices = reference_values.index.intersection(values_to_scale.index)

    # if we are going to weight the scaling using the uncertainty values, then the weights will be
    #    inverse_sigma = 1 / sqrt{ sigmaA ** 2 + sigmaB ** 2 }
    if reference_uncertainties is not None and to_scale_uncertainties is not None:
        assert reference_uncertainties.index.equals(reference_values.index)
        assert to_scale_uncertainties.index.equals(values_to_scale.index)
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
    assert len(optimized_scale_factors) == len(values_to_scale)

    return optimized_scale_factors


def scale_datasets(
    reference_dataset: rs.DataSet,
    dataset_to_scale: rs.DataSet,
    column_to_scale: str = "F",
    uncertainty_column: str = "SIGF",
    weight_using_uncertainties: bool = True,
) -> rs.DataSet:
    """
    Apply an anisotropic scaling so that `dataset_to_scale` is on the same scale as `reference`.

        C * exp{ -(h**2 B11 + k**2 B22 + l**2 B33 +
                    2hk B12 + 2hl  B13 +  2kl B23) }

    This is the same procedure implemented by CCP4's SCALEIT.

    Assumes that the `reference` and `dataset_to_scale` both have a column `column_to_scale`.
    """

    if weight_using_uncertainties:
        scale_factors = compute_scale_factors(
            reference_values=reference_dataset[column_to_scale],
            values_to_scale=dataset_to_scale[column_to_scale],
            reference_uncertainties=reference_dataset[uncertainty_column],
            to_scale_uncertainties=dataset_to_scale[uncertainty_column],
        )
    else:
        scale_factors = compute_scale_factors(
            reference_values=reference_dataset[column_to_scale],
            values_to_scale=dataset_to_scale[column_to_scale],
            reference_uncertainties=None,
            to_scale_uncertainties=None,
        )

    scaled_dataset = dataset_to_scale.copy()
    columns_to_scale = [col for col in dataset_to_scale.columns if type(col) in DTYPES_TO_SCALE]
    for column in columns_to_scale:
        scaled_dataset[column] *= scale_factors

    return scaled_dataset
