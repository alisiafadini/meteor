from typing import Literal, overload

import numpy as np
import reciprocalspaceship as rs
import scipy.optimize as opt


@overload
def scale_structure_factors(
    reference: rs.DataSeries,
    dataset_to_scale: rs.DataSeries,
    inplace: Literal[True] = True,
) -> None:
    ...


@overload
def scale_structure_factors(
    reference: rs.DataSeries,
    dataset_to_scale: rs.DataSeries,
    inplace: Literal[False] = False,
) -> rs.DataSeries:
    ...


def scale_structure_factors(
    reference: rs.DataSeries,
    dataset_to_scale: rs.DataSeries,
    inplace: bool = True,
) -> None | rs.DataSeries:
    """
    Apply an anisotropic scaling so that `dataset_to_scale` is on the same scale as `reference`.

    C * exp{ -(h**2 B11 + k**2 B22 + l**2 B33 +
                2hk B12 + 2hl  B13 +  2kl B23) }

    This is the same procedure implemented by CCP4's SCALEIT.

    Parameters:
    reference (rs.DataSeries): Single-column DataSeries to use as the reference for scaling.
    dataset_to_scale (rs.DataSeries): Single-column DataSeries to be scaled.
    inplace (bool): If `True`, modifies the original DataSeries. If `False`,
    returns a new scaled DataSeries.

    Returns:
    None if `inplace` is True, otherwise rs.DataSeries with scaled data.
    """

    def aniso_scale_func(params, x_ref, x_scale, miller_indices):
        h, k, l = miller_indices[:, 0], miller_indices[:, 1], miller_indices[:, 2]  # noqa: E741
        h_sq, k_sq, l_sq = np.square(h), np.square(k), np.square(l)
        hk_prod, hl_prod, kl_prod = h * k, h * l, k * l

        t = -(
            h_sq * params[1]
            + k_sq * params[2]
            + l_sq * params[3]
            + 2 * hk_prod * params[4]
            + 2 * hl_prod * params[5]
            + 2 * kl_prod * params[6]
        )

        return x_ref - params[0] * np.exp(t) * x_scale

    reference_data = reference.to_numpy()
    scale_data = dataset_to_scale.to_numpy()

    miller_indices_ref = np.array(list(reference.index))
    miller_indices_scale = np.array(list(dataset_to_scale.index))

    assert np.array_equal(
        miller_indices_ref, miller_indices_scale
    ), "Miller indices of reference and dataset_to_scale do not match."  # noqa: E501

    # Initial guess for: [C, B11, B22, B33, B12, B13, B23]
    initial_params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    result = opt.least_squares(
        aniso_scale_func,
        initial_params,
        args=(reference_data, scale_data, miller_indices_scale),
    )

    # Apply the scaling to dataset_to_scale
    h, k, l = miller_indices_scale[:, 0], miller_indices_scale[:, 1], miller_indices_scale[:, 2]  # noqa: E741
    h_sq, k_sq, l_sq = np.square(h), np.square(k), np.square(l)
    hk_prod, hl_prod, kl_prod = h * k, h * l, k * l

    t = -(
        h_sq * result.x[1]
        + k_sq * result.x[2]
        + l_sq * result.x[3]
        + 2 * hk_prod * result.x[4]
        + 2 * hl_prod * result.x[5]
        + 2 * kl_prod * result.x[6]
    )

    scaled_data = (result.x[0] * np.exp(t)) * scale_data

    if inplace:
        dataset_to_scale[:] = scaled_data
        return None
    else:
        scaled_dataset = dataset_to_scale.copy()
        scaled_dataset[:] = scaled_data
        return scaled_dataset

def compute_amplitude_fofo_difference(
    data1: rs.DataSeries, data2: rs.DataSeries, data3: rs.DataSeries
) -> rs.DataSeries:
    """
    First, scale data1 and data2 to the common scale defined by data3
    then compute the difference (data2 - data1).

    Parameters:
    data1 (rs.DataSeries): First dataset to be used in difference calculation (e.g. F_off).
    data2 (rs.DataSeries): Second dataset to be used in difference calculation (e.g. F_on).
    data3 (rs.DataSeries): Reference dataset used for scaling data1 and data2 (e.g. F_calc).

    Returns:
    rs.DataSeries: The difference (data2 - data1) after scaling to the reference scale.
    """

    # Scale data1 and data2 to the scale of data3
    scaled_data1 = scale_structure_factors(reference=data3, dataset_to_scale=data1, inplace=False)
    scaled_data2 = scale_structure_factors(reference=data3, dataset_to_scale=data2, inplace=False)

    # Compute the difference between the scaled data2 and data1
    difference = scaled_data2 - scaled_data1

    return difference
