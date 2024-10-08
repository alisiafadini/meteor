import numpy as np
import reciprocalspaceship as rs

from .tv import tv_denoise_difference_map
from .utils import canonicalize_amplitudes


def _form_complex_sf(amplitudes: rs.DataSeries, phases_in_deg: rs.DataSeries) -> np.ndarray:
    expi = lambda x: np.exp(1j * np.deg2rad(x))  # noqa: E731
    return amplitudes.to_numpy().astype(np.complex128) * expi(phases_in_deg.to_numpy().astype(np.float64))


def _complex_argument(complex: rs.DataSeries) -> rs.DataSeries:
    return complex.apply(np.angle).apply(np.rad2deg).astype(rs.PhaseDtype())


def _dataseries_l1_norm(
    series1: rs.DataSeries,
    series2: rs.DataSeries,
) -> float:
    difference = (series2 - series1).dropna()
    num_datapoints = len(difference)
    if num_datapoints == 0:
        raise RuntimeError("no overlapping indices between `series1` and `series2`")
    return np.sum(np.abs(difference)) / float(num_datapoints)


def _projected_derivative_phase(
    *,
    difference_amplitudes: rs.DataSeries,
    difference_phases: rs.DataSeries,
    native_amplitudes: rs.DataSeries,
    native_phases: rs.DataSeries,
) -> rs.DataSeries:
    complex_difference = _form_complex_sf(difference_amplitudes, difference_phases)
    complex_native = _form_complex_sf(native_amplitudes, native_phases)
    complex_derivative_estimate = complex_difference + complex_native
    complex_derivative_estimate = rs.DataSeries(complex_derivative_estimate, index=native_amplitudes.index)
    print(complex_derivative_estimate)
    return _complex_argument(complex_derivative_estimate)


def iterative_tv_phase_retrieval(
    input_dataset: rs.DataSet,
    *,
    native_amplitude_column: str = "F",
    derivative_amplitude_column: str = "Fh",
    calculated_phase_column: str = "PHIC",
    output_derivative_phase_column: str = "PHICh",
    convergence_tolerance: float = 0.01,
) -> rs.DataSet:
    """
    Here is a brief psuedocode sketch of the alogrithm. Structure factors F below are complex unless
    explicitly annotated |*|.

        Input: |F|, |Fh|, phi_c
        Note: F = |F| * exp{ phi_c } is the native/dark data,
             |Fh| represents the derivative/triggered/light data

        Initialize:
         - D_F = ( |Fh| - |F| ) * exp{ phi_c }

        while not converged:
            D_rho = FT{ D_F }                       Fourier transform
            D_rho' = TV{ D_rho }                    TV denoise: apply real space prior
            D_F' = FT-1{ D_rho' }                   back Fourier transform
            Fh' = (D_F' + F) * [|Fh| / |D_F' + F|]  Fourier space projection onto experimental set
            D_F = Fh' - F

    Where the TV lambda parameter is determined using golden section optimization. The algorithm
    iterates until the changes in DF for each iteration drop below a specified per-amplitude
    threshold.
    """

    # TODO work on below for readability
    # TODO should these be adjustable input params? not returned?
    difference_amplitude_column: str = "DF"
    difference_phase_column: str = "DPHIC"

    # input_dataset = input_dataset.copy()

    initial_derivative_phases = input_dataset[calculated_phase_column].rename(
        output_derivative_phase_column
    )
    initial_difference_amplitudes = (
        input_dataset[derivative_amplitude_column] - input_dataset[native_amplitude_column]
    ).rename(difference_amplitude_column)

    initial_difference_phases = input_dataset[calculated_phase_column].rename(
        difference_phase_column
    )

    # working_ds holds 3x complex numbers: native, derivative, differences
    working_ds: rs.DataSet = rs.concat(
        [
            input_dataset[native_amplitude_column],
            input_dataset[derivative_amplitude_column],
            input_dataset[calculated_phase_column],
            initial_derivative_phases,
            initial_difference_amplitudes,
            initial_difference_phases,
        ],
        axis=1,
        check_isomorphous=False,
    )
    working_ds.spacegroup = input_dataset.spacegroup
    working_ds.cell = input_dataset.cell

    # begin iterative TV algorithm
    converged: bool = False

    while not converged:
        # TV denoise using golden algorithm, includes forward and backwards FTs
        # this is the un-projected difference amplitude and phase
        # > returned DF_prime is a copy with new `DIFFERENCE_AMPLITUDES` & `calculated_phase_column`
        DF_prime, result = tv_denoise_difference_map(  # noqa: N806
            difference_map_coefficients=working_ds,
            difference_map_amplitude_column=difference_amplitude_column,
            difference_map_phase_column=difference_phase_column,
            lambda_values_to_scan=[
                0.1,
            ],
            full_output=True,
        )

        change_in_DF = _dataseries_l1_norm(  # noqa: N806
            working_ds[difference_amplitude_column],  # previous iteration
            DF_prime[difference_amplitude_column],  # current iteration
        )
        converged = change_in_DF < convergence_tolerance
        print("***", result.optimal_negentropy, change_in_DF)

        # update working_ds, NB native and derivative amplitudes & native phases stay the same
        working_ds[output_derivative_phase_column] = _projected_derivative_phase(
            difference_amplitudes=DF_prime[difference_amplitude_column],
            difference_phases=DF_prime[difference_phase_column],
            native_amplitudes=working_ds[native_amplitude_column],
            native_phases=working_ds[calculated_phase_column],
        )

        # TODO encapsulate block below into function
        current_complex_native = _form_complex_sf(
            working_ds[native_amplitude_column], working_ds[calculated_phase_column]
        )
        current_complex_derivative = _form_complex_sf(
            working_ds[derivative_amplitude_column], working_ds[output_derivative_phase_column]
        )

        current_complex_difference = current_complex_derivative - current_complex_native
        working_ds[difference_amplitude_column] = rs.DataSeries(
            np.abs(current_complex_difference),
            index=working_ds.index,
            name=difference_amplitude_column
        ).astype(rs.StructureFactorAmplitudeDtype())
        working_ds[difference_phase_column] = _complex_argument(current_complex_difference)

    canonicalize_amplitudes(
        working_ds,
        amplitude_label=derivative_amplitude_column,
        phase_label=output_derivative_phase_column,
        inplace=True,
    )
    canonicalize_amplitudes(
        working_ds,
        amplitude_label=difference_amplitude_column,
        phase_label=difference_phase_column,
        inplace=True,
    )

    return working_ds
