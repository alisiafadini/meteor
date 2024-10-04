from typing import Literal, Optional, Tuple, Union, overload

import numpy as np
import reciprocalspaceship as rs
import scipy.optimize as opt


@overload
def scale_structure_factors(
    reference: rs.DataSeries,
    dataset_to_scale: rs.DataSeries,
    uncertainties: Optional[rs.DataSeries] = None,
    inplace: Literal[True] = True,
) -> None: ...


@overload
def scale_structure_factors(
    reference: rs.DataSeries,
    dataset_to_scale: rs.DataSeries,
    uncertainties: Optional[rs.DataSeries] = None,
    inplace: Literal[False] = False,
) -> Union[rs.DataSeries, tuple[rs.DataSeries, rs.DataSeries]]: ...


def scale_structure_factors(
    reference: rs.DataSeries,
    dataset_to_scale: rs.DataSeries,
    uncertainties: Optional[rs.DataSeries] = None,
    inplace: bool = True,
) -> None | rs.DataSeries | Tuple[rs.DataSeries, rs.DataSeries]:
    """
    Apply an anisotropic scaling so that `dataset_to_scale` is on the same scale as `reference`.

    C * exp{ -(h**2 B11 + k**2 B22 + l**2 B33 +
                2hk B12 + 2hl  B13 +  2kl B23) }

    This is the same procedure implemented by CCP4's SCALEIT.

    Parameters:
    reference (rs.DataSeries): Single-column DataSeries to use as the reference for scaling.
    dataset_to_scale (rs.DataSeries): Single-column DataSeries to be scaled.
    uncertainties (Optional[rs.DataSeries]): Optional uncertainties associated with
    the dataset to scale.
    inplace (bool): If `True`, modifies the original DataSeries. If `False`,
    returns a new scaled DataSeries.

    Returns:
    None if `inplace` is True, otherwise a scaled rs.DataSeries (and optionally uncertainties).
    """

    def aniso_scale_objective(
        scale_params: np.ndarray,
        reference_data: np.ndarray,
        data_to_scale: np.ndarray,
        miller_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Objective function to scale two data arrays using an anisotropic scaling model.

        Parameters:
        scale_params (np.ndarray): Array of scaling parameters [C, B11, B22, B33, B12, B13, B23].
        reference_data (np.ndarray): Array of reference data.
        data_to_scale (np.ndarray): Array of data to scale.
        miller_indices (np.ndarray): Array of Miller indices, with columns for h, k, l.

        Returns:
        np.ndarray: Residual between scaled data and reference data.
        """
        h, k, l = miller_indices[:, 0], miller_indices[:, 1], miller_indices[:, 2]
        h_sq, k_sq, l_sq = np.square(h), np.square(k), np.square(l)
        hk_prod, hl_prod, kl_prod = h * k, h * l, k * l

        # Anisotropic scaling term
        exponent_term = -(
            h_sq * scale_params[1]
            + k_sq * scale_params[2]
            + l_sq * scale_params[3]
            + 2 * hk_prod * scale_params[4]
            + 2 * hl_prod * scale_params[5]
            + 2 * kl_prod * scale_params[6]
        )

        scaled_data = scale_params[0] * np.exp(exponent_term) * data_to_scale

        # Residual between scaled data and reference data
        return scaled_data - reference_data

    # Convert DataSeries to numpy arrays
    reference_data = reference.to_numpy()
    scale_data = dataset_to_scale.to_numpy()
    uncertainties_data = uncertainties.to_numpy() if uncertainties is not None else None

    # Miller indices (list casting for multi index)
    miller_indices_ref = np.array(list(reference.index))
    miller_indices_scale = np.array(list(dataset_to_scale.index))

    # Ensure Miller indices match between the two datasets
    assert np.array_equal(
        miller_indices_ref, miller_indices_scale
    ), "Miller indices of reference and dataset_to_scale do not match."

    # Initial guess for the scaling parameters: [C, B11, B22, B33, B12, B13, B23]
    initial_params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Least-squares optimization to fit scaling parameters
    result = opt.least_squares(
        aniso_scale_objective,
        initial_params,
        args=(reference_data, scale_data, miller_indices_scale),
    )

    # Extract optimized scaling parameters
    optimized_params = result.x

    # Recalculate the scaling factor and apply it to dataset.
    h, k, l = (
        miller_indices_scale[:, 0],
        miller_indices_scale[:, 1],
        miller_indices_scale[:, 2],
    )
    h_sq, k_sq, l_sq = np.square(h), np.square(k), np.square(l)
    hk_prod, hl_prod, kl_prod = h * k, h * l, k * l

    exponent_term = -(
        h_sq * optimized_params[1]
        + k_sq * optimized_params[2]
        + l_sq * optimized_params[3]
        + 2 * hk_prod * optimized_params[4]
        + 2 * hl_prod * optimized_params[5]
        + 2 * kl_prod * optimized_params[6]
    )

    scale_factor = optimized_params[0] * np.exp(exponent_term)
    scaled_data = scale_factor * scale_data

    # Apply scaling to uncertainties if they are provided
    if uncertainties is not None:
        scaled_uncertainties = scale_factor * uncertainties_data
    else:
        scaled_uncertainties = None

    # Modify the dataset in-place or return a new scaled dataset
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
