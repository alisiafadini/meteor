"""iterative TV-based phase retrieval"""

from __future__ import annotations

import numpy as np
import pandas as pd
import reciprocalspaceship as rs
import structlog
from reciprocalspaceship.decorators import cellify, spacegroupify

from .rsmap import Map
from .settings import (
    DEFAULT_TV_WEIGHTS_TO_SCAN_AT_EACH_ITERATION,
    ITERATIVE_TV_CONVERGENCE_TOLERANCE,
    ITERATIVE_TV_MAX_ITERATIONS,
)
from .tv import TvDenoiseResult, tv_denoise_difference_map
from .utils import CellType, SpacegroupType, average_phase_diff_in_degrees, filter_common_indices

log = structlog.get_logger()


class _IterativeTvDenoiser:
    @cellify("cell")
    @spacegroupify("spacegroup")
    def __init__(  # noqa: PLR0913
        self,
        *,
        cell: CellType,
        spacegroup: SpacegroupType,
        convergence_tolerance: float = ITERATIVE_TV_CONVERGENCE_TOLERANCE,
        max_iterations: int = ITERATIVE_TV_MAX_ITERATIONS,
        tv_weights_to_scan: list[float] = DEFAULT_TV_WEIGHTS_TO_SCAN_AT_EACH_ITERATION,
        verbose: bool = False,
    ) -> None:
        """
        Initialize a TV denoiser.

        Parameters
        ----------
        cell: gemmi.Cell | Sequence[float] | np.ndarray
            Unit cell, should match both the `native` and `derivative` datasets (usual: use native)

        spacegroup: gemmi.SpaceGroup | str | int
            The spacegroup; both the `native` and `derivative` datasets

        convergance_tolerance: float
            If the change in the estimated derivative SFs drops below this value (phase,
            per-component) then return. Default 1e-4.

        max_iterations: int
            If this number of iterations is reached, stop early. Default 1000.

        tv_weights_to_scan : list[float], optional
            A list of TV regularization weights (λ values) to be scanned for optimal results,
            by default [0.001, 0.01, 0.1, 1.0].

        verbose: bool
            Log or not.
        """
        self.cell = cell
        self.spacegroup = spacegroup
        self.tv_weights_to_scan = tv_weights_to_scan
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose

        if verbose:
            log.info(
                "convergence criteria:",
                phase_tolerance=convergence_tolerance,
                max_iterations=max_iterations,
            )

    def _tv_denoise_complex_difference_sf(
        self, complex_difference_sf: rs.DataSeries
    ) -> tuple[rs.DataSeries, TvDenoiseResult]:
        """Apply a single iteration of TV denoising to set of complex SFs, return complex SFs"""
        diffmap = Map.from_structurefactor(
            complex_difference_sf,
            index=complex_difference_sf.index,
            cell=self.cell,
            spacegroup=self.spacegroup,
        )

        denoised_map, tv_metadata = tv_denoise_difference_map(
            diffmap,
            weights_to_scan=self.tv_weights_to_scan,
            full_output=True,
        )

        return denoised_map.to_structurefactor(), tv_metadata

    def __call__(
        self,
        *,
        native: rs.DataSeries,
        initial_derivative: rs.DataSeries,
    ) -> tuple[rs.DataSeries, pd.DataFrame]:
        """
        Estimate the derivative phases using the iterative TV algorithm.

        This function contains the algorithm logic.

        Parameters
        ----------
        native: rs.DataSeries
            The complex native structure factors, usually experimental amplitudes and calculated phases

        initial_complex_derivative : rs.DataSeries
            The complex derivative structure factors, usually with experimental amplitudes and esimated
            phases (often calculated from the native structure)

        Returns
        -------
        estimated_complex_derivative: rs.DataSeries
            The derivative SFs, with the same amplitudes but phases altered to minimize the TV.

        metadata: pd.DataFrame
            Information about the algorithm run as a function of iteration. For each step, includes:
            the tv_weight used, the negentropy (after the TV step), and the average phase change in
            degrees.
        """
        if not isinstance(native, rs.DataSeries):
            msg = f"`native` must be type rs.DataSeries, got {type(native)}"
            raise TypeError(msg)

        if not isinstance(initial_derivative, rs.DataSeries):
            msg = f"`initial_derivative` must be type rs.DataSeries, got {type(initial_derivative)}"
            raise TypeError(msg)

        derivative = initial_derivative.copy()
        converged: bool = False
        num_iterations: int = 0
        metadata: list[dict[str, float]] = []

        # do differences with rs.DataSeries, handles missing indices
        difference: rs.DataSeries = initial_derivative - native

        while not converged:
            denoised_difference, tv_metadata = self._tv_denoise_complex_difference_sf(difference)

            # project onto the native amplitudes to obtain an "updated_derivative"
            #   Fh' = (D_F' + F) * [|Fh| / |D_F' + F|]
            updated_derivative: rs.DataSeries = denoised_difference + native
            updated_derivative *= np.abs(derivative) / np.abs(updated_derivative)

            # compute phase change, THEN set: derivative <- updated_derivative
            phase_change = average_phase_diff_in_degrees(derivative, updated_derivative)
            derivative = updated_derivative

            difference = derivative - native

            converged = phase_change < self.convergence_tolerance
            num_iterations += 1

            metadata.append(
                {
                    "iteration": num_iterations,
                    "tv_weight": tv_metadata.optimal_tv_weight,
                    "negentropy_after_tv": tv_metadata.optimal_negentropy,
                    "average_phase_change": phase_change,
                },
            )
            if self.verbose:
                log.info(
                    f"  iteration {num_iterations:04d}",  # noqa: G004
                    phase_change=round(phase_change, 4),
                    negentropy=round(tv_metadata.optimal_negentropy, 4),
                    tv_weight=tv_metadata.optimal_tv_weight,
                )

            if num_iterations > self.max_iterations:
                break

        return derivative, pd.DataFrame(metadata)


def iterative_tv_phase_retrieval(  # noqa: PLR0913
    initial_derivative: Map,
    native: Map,
    *,
    convergence_tolerance: float = ITERATIVE_TV_CONVERGENCE_TOLERANCE,
    max_iterations: int = ITERATIVE_TV_MAX_ITERATIONS,
    tv_weights_to_scan: list[float] = DEFAULT_TV_WEIGHTS_TO_SCAN_AT_EACH_ITERATION,
    verbose: bool = False,
) -> tuple[Map, pd.DataFrame]:
    """
    Here is a brief pseudocode sketch of the alogrithm. Structure factors F below are complex unless
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

    Where the TV weight parameter is determined using golden section optimization. The algorithm
    iterates until the changes in the derivative phase drop below a specified threshold.

    Parameters
    ----------
    initial_derivative: Map
        the derivative amplitudes, and initial guess for the phases

    native: Map
        the native amplitudes, phases

    convergance_tolerance: float
        If the change in the estimated derivative SFs drops below this value (phase, per-component)
        then return. Default 1e-4.

    max_iterations: int
        If this number of iterations is reached, stop early. Default 1000.

    tv_weights_to_scan : list[float], optional
        A list of TV regularization weights (λ values) to be scanned for optimal results,
        by default [0.001, 0.01, 0.1, 1.0].

    verbose: bool
        Log or not.

    Returns
    -------
    output_map: Map
        The estimated derivative phases, along with the input amplitudes and input computed phases.

    metadata: pd.DataFrame
        Information about the algorithm run as a function of iteration. For each step, includes:
        the tv_weight used, the negentropy (after the TV step), and the average phase change in
        degrees.
    """
    initial_derivative, native = filter_common_indices(initial_derivative, native)

    denoiser = _IterativeTvDenoiser(
        cell=native.cell,
        spacegroup=native.spacegroup,
        convergence_tolerance=convergence_tolerance,
        max_iterations=max_iterations,
        tv_weights_to_scan=tv_weights_to_scan,
        verbose=verbose,
    )

    it_tv_complex_derivative, metadata = denoiser(
        native=native.to_structurefactor(),
        initial_derivative=initial_derivative.to_structurefactor(),
    )

    updated_derivative_map = Map.from_structurefactor(
        it_tv_complex_derivative,
        cell=initial_derivative.cell,
        spacegroup=initial_derivative.spacegroup,
    )

    if initial_derivative.has_uncertainties:
        updated_derivative_map.set_uncertainties(
            initial_derivative.uncertainties,
            column_name=initial_derivative.uncertainties_column_name,
        )

    return updated_derivative_map, metadata
