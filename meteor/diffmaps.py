import numpy as np
import pandas as pd
import reciprocalspaceship as rs

from .settings import TV_MAP_SAMPLING
from .utils import compute_map_from_coefficients
from .validate import ScalarMaximizer, negentropy


def _rs_dataseries_to_complex_array(
    amplitudes: rs.DataSeries,
    phases: rs.DataSeries,
) -> np.ndarray:
    pd.testing.assert_index_equal(
        amplitudes.index, phases.index
    )  # Ensure indices match
    # Convert amplitudes and phases to complex representation
    return amplitudes.to_numpy() * np.exp(1j * np.deg2rad(phases.to_numpy()))


def _complex_array_to_rs_dataseries(
    complex_array: np.ndarray,
    index: pd.Index,
) -> tuple[rs.DataSeries, rs.DataSeries]:
    # Extract amplitude from the complex array (magnitude)
    amplitudes = rs.DataSeries(np.abs(complex_array), index=index)
    amplitudes = amplitudes.astype(rs.StructureFactorAmplitudeDtype())  # Convert dtype

    # Extract phase from the complex array (angle, in degrees)
    phases = rs.DataSeries(np.rad2deg(np.angle(complex_array)), index=index)
    phases = phases.astype(rs.PhaseDtype())  # Convert dtype

    return amplitudes, phases


def compute_deltafofo(
    dataset: rs.DataSet,
    *,
    native_amplitudes: str,
    derivative_amplitudes: str,
    native_phases: str,
    derivative_phases: str | None = None,
    inplace: bool = False,
) -> rs.DataSet | None:
    """
    Compute amplitude and phase differences between native and derivative datasets.
    """

    if not inplace:
        dataset = dataset.copy()

    # Convert native and derivative amplitude/phase pairs to complex arrays
    native_complex = _rs_dataseries_to_complex_array(
        dataset[native_amplitudes], dataset[native_phases]
    )

    if derivative_phases is not None:
        derivative_complex = _rs_dataseries_to_complex_array(
            dataset[derivative_amplitudes], dataset[derivative_phases]
        )
    else:
        # If no derivative phases are provided, assume they are the same as native phases
        derivative_complex = _rs_dataseries_to_complex_array(
            dataset[derivative_amplitudes], dataset[native_phases]
        )

    # Compute complex differences
    delta_complex = derivative_complex - native_complex

    # Convert back to amplitude and phase DataSeries
    delta_amplitudes, delta_phases = _complex_array_to_rs_dataseries(
        delta_complex, dataset.index
    )

    # Add results to dataset
    dataset["DeltaFoFo"] = delta_amplitudes
    dataset["DeltaPhases"] = delta_phases

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
    native_phases: str,
    derivative_phases: str | None = None,
    sigf_native: str,
    sigf_deriv: str,
    kweight: float = 1.0,
    optimize_kweight: bool = False,
    inplace: bool = False,
) -> rs.DataSet | None:
    """
    Compute k-weighted differences between native and derivative amplitudes and phases.

    Assumes that scaling has already been applied to the amplitudes before calling this function.

    Parameters:
    -----------
    dataset : rs.DataSet
        The input dataset containing columns for native and derivative amplitudes/phases.
    native_amplitudes : str
        Column label for native amplitudes in the dataset.
    derivative_amplitudes : str
        Column label for derivative amplitudes in the dataset.
    native_phases : str, optional
        Column label for native phases.
    derivative_phases : str, optional
        Column label for derivative phases, by default None.
    sigf_native : str
        Column label for uncertainties of native amplitudes.
    sigf_deriv : str
        Column label for uncertainties of derivative amplitudes.
    kweight : float, optional
        k-weight factor, by default 1.0.
    optimize_kweight : bool, optional
        Whether to optimize the kweight using negentropy, by default False.
    inplace : bool, optional
        Whether to modify the dataset in place. Default is False.

    Returns:
    --------
    rs.DataSet | None
        The modified dataset with k-weighted differences, if inplace=False, otherwise None.
    """
    if not inplace:
        dataset = dataset.copy()

    # Compute differences between native and derivative amplitudes and phases
    dataset = compute_deltafofo(
        dataset=dataset,
        native_amplitudes=native_amplitudes,
        derivative_amplitudes=derivative_amplitudes,
        native_phases=native_phases,
        derivative_phases=derivative_phases,
        inplace=inplace,
    )

    delta_amplitudes = dataset["DeltaFoFo"]
    sigdelta_amplitudes = np.sqrt(dataset[sigf_deriv] ** 2 + dataset[sigf_native] ** 2)

    if optimize_kweight:

        def negentropy_objective(kweight_value: float) -> float:
            # Apply k-weighting to DeltaFoFo
            weights = compute_kweights(
                delta_amplitudes, sigdelta_amplitudes, kweight_value
            )
            weighted_delta_fofo = delta_amplitudes * weights
            dataset["DeltaFoFoKWeighted"] = weighted_delta_fofo

            # Convert weighted amplitudes and phases to a map
            delta_map = compute_map_from_coefficients(
                map_coefficients=dataset,
                amplitude_label="DeltaFoFoKWeighted",
                phase_label=native_phases,
                map_sampling=TV_MAP_SAMPLING,
            )

            delta_map_as_array = np.array(delta_map.grid)

            # Compute negentropy of the map
            return negentropy(delta_map_as_array.flatten())

        # Optimize kweight using negentropy objective
        maximizer = ScalarMaximizer(objective=negentropy_objective)
        maximizer.optimize_with_golden_algorithm(bracket=(0.1, 10.0))
        kweight = maximizer.argument_optimum

    # Compute weights and apply to DeltaFoFo
    weights = compute_kweights(delta_amplitudes, sigdelta_amplitudes, kweight)
    dataset["DeltaFoFoKWeighted"] = delta_amplitudes * weights

    if inplace:
        return None
    else:
        return dataset
