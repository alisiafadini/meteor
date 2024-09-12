from typing import Literal, overload, Union, Optional

import gemmi
import numpy as np
import reciprocalspaceship as rs
import scipy.optimize as opt


@overload
def scale_structure_factors(
    reference: rs.DataSeries,
    dataset_to_scale: rs.DataSeries,
    uncertainties: Optional[rs.DataSeries] = None,
    inplace: Literal[True] = True,
) -> None:
    ...


@overload
def scale_structure_factors(
    reference: rs.DataSeries,
    dataset_to_scale: rs.DataSeries,
    uncertainties: Optional[rs.DataSeries] = None,
    inplace: Literal[False] = False,
) -> Union[rs.DataSeries, tuple[rs.DataSeries, rs.DataSeries]]:
    ...


def scale_structure_factors(
    reference: rs.DataSeries,
    dataset_to_scale: rs.DataSeries,
    uncertainties: Optional[rs.DataSeries] = None,
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
    uncertainties_data = uncertainties.to_numpy() if uncertainties is not None else None

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

    scale_factor = result.x[0] * np.exp(t)
    scaled_data = scale_factor * scale_data

    if uncertainties is not None:
        scaled_uncertainties = scale_factor * uncertainties_data
    else:
        scaled_uncertainties = None

    if inplace:
        dataset_to_scale[:] = scaled_data
        if uncertainties is not None:
            uncertainties[:] = scaled_uncertainties
        return None
    else:
        scaled_dataset = dataset_to_scale.copy()
        scaled_dataset[:] = scaled_data
        if uncertainties is not None:
            scaled_uncertainties_series = uncertainties.copy()
            scaled_uncertainties_series[:] = scaled_uncertainties
            return scaled_dataset, scaled_uncertainties_series
        return scaled_dataset

def compute_amplitude_fofo_difference(
    data1: rs.DataSeries,
    data2: rs.DataSeries,
    data3: rs.DataSeries,
    uncertainties1: Optional[rs.DataSeries] = None,
    uncertainties2: Optional[rs.DataSeries] = None,
) -> Union[rs.DataSeries, tuple[rs.DataSeries, rs.DataSeries]]:
    """
    First, scale data1 and data2 to the common scale defined by data3,
    then compute the difference (data2 - data1). Optionally, propagate uncertainties.

    Parameters:
    data1 (rs.DataSeries): First dataset to be used in difference calculation (e.g., F_off).
    data2 (rs.DataSeries): Second dataset to be used in difference calculation (e.g., F_on).
    data3 (rs.DataSeries): Reference dataset used for scaling data1 and data2 (e.g., F_calc).
    uncertainties1 (Optional[rs.DataSeries]): Uncertainties corresponding to data1 (e.g. SIGF_off).
    uncertainties2 (Optional[rs.DataSeries]): Uncertainties corresponding to data2 (e.g. SIGF_on).

    Returns:
    rs.DataSeries or tuple: The difference (data2 - data1) after scaling to the reference scale.
    If uncertainties are provided, returns a tuple (difference, propagated_uncertainties).
    """

    # Ensure that both or neither uncertainties are provided
    if (uncertainties1 is not None) != (uncertainties2 is not None):
        raise ValueError("Either provide uncertainties for both data1 and data2, or provide none.")

    if uncertainties1 is not None and uncertainties2 is not None:
        # Scale data and uncertainties
        scaled_data1, scaled_uncertainties1 = scale_structure_factors(
            reference=data3, dataset_to_scale=data1, uncertainties=uncertainties1, inplace=False
        )
        scaled_data2, scaled_uncertainties2 = scale_structure_factors(
            reference=data3, dataset_to_scale=data2, uncertainties=uncertainties2, inplace=False
        )
        # Compute the difference between the scaled data2 and data1
        difference = scaled_data2 - scaled_data1
        # Propagate the uncertainties (assuming independent errors)
        propagated_uncertainties = np.sqrt(scaled_uncertainties1**2 + scaled_uncertainties2**2)
        return difference, propagated_uncertainties
    else:
        # Scale data without uncertainties
        scaled_data1 = scale_structure_factors(reference=data3,
                                               dataset_to_scale=data1,
                                               inplace=False)
        scaled_data2 = scale_structure_factors(reference=data3,
                                               dataset_to_scale=data2,
                                               inplace=False)
        # Compute the difference between the scaled data2 and data1
        difference = scaled_data2 - scaled_data1
        return difference


def compute_fofo_difference_map(
    data1: rs.DataSeries,
    data2: rs.DataSeries,
    f_calcs: rs.DataSeries,
    phases: rs.DataSeries,
    spacegroup: gemmi.SpaceGroup,
    cell: gemmi.UnitCell
) -> rs.DataSet:
    """
    Compute the FoFo difference map using two datasets and pre-calculated structure factors.

    Parameters:
    data1 (rs.DataSeries): First dataset for FoFo difference computation.
    data2 (rs.DataSeries): Second dataset for FoFo difference computation.
    f_calcs (rs.DataSeries): Pre-calculated structure factor amplitudes.
    phases (rs.DataSeries): Pre-calculated phases.

    Returns:
    rs.DataSet: A DataSet containing the pre-calculated phases and the FoFo amplitude differences.
    """

    # Compute the FoFo amplitude differences by scaling to fcalcs
    fofo_diff = compute_amplitude_fofo_difference(data1, data2, f_calcs)

    # Create the final DataSet with phases and FoFo differences
    result = rs.DataSet({
        "FoFo": fofo_diff,
        "Phases": phases,
    }) #TODO standardize outputs in settings.py?

    #Assign unit cell, spacegroup
    result.spacegroup = spacegroup
    result.cell = cell

    return result




