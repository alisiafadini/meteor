import numpy as np
import reciprocalspaceship as rs

from .settings import TV_MAP_SAMPLING
from .utils import compute_map_from_coefficients
from .validate import ScalarMaximizer, negentropy


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
        Column label for derivative phases. Defaults to None, in which case native phases are used.
    inplace : bool, optional
        Whether to modify the dataset in place. Default is False.

    Returns:
    --------
    rs.DataSet | None
        The modified dataset with differences if inplace=False, otherwise None.
    """
    if not inplace:
        dataset = dataset.copy()

    # Compute amplitude differences (no scaling assumed here)
    delta_amplitudes = dataset[derivative_amplitudes] - dataset[native_amplitudes]
    dataset["DeltaFoFo"] = delta_amplitudes

    # Handle phase differences
    if native_phases is not None and derivative_phases is not None:
        native_phase = dataset[native_phases].to_numpy(
            dtype=np.float32
        )  # Convert PhaseArray to float32 for phase difference below
        derivative_phase = dataset[derivative_phases].to_numpy(dtype=np.float32)
    elif native_phases is not None:
        native_phase = dataset[native_phases].to_numpy(dtype=np.float32)
        derivative_phase = native_phase
    else:
        raise ValueError("At least native_phases must be provided.")

    # Compute phase differences (angle normalization between -pi and pi)
    delta_phases = np.angle(np.exp(1j * (derivative_phase - native_phase)))

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
