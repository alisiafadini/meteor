from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import structlog

from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    max_negentropy_kweighted_difference_map,
)
from meteor.rsmap import Map
from meteor.settings import TV_WEIGHT_DEFAULT
from meteor.tv import TvDenoiseResult, tv_denoise_difference_map

from .common import DiffmapArgParser, DiffMapSet, InvalidWeightModeError, WeightMode

log = structlog.get_logger()


# TODO: optimize
TV_LAMBDA_VALUES_TO_SCAN = np.linspace(0.0, 0.1, 10)


class TvDiffmapArgParser(DiffmapArgParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_argument(
            "-tv",
            "--tv-denoise-mode",
            type=str,
            default=WeightMode.optimize,
            choices=WeightMode,
            help="Choose how to find a TV denoising regularization weight (lambda) parameter.",
        )
        self.add_argument(
            "-l",
            "--tv-weight",
            type=float,
            default=TV_WEIGHT_DEFAULT,
            help=(
                f"If `--tv-denoise-mode {WeightMode.fixed}`, set the TV weight parameter to this "
                f"value. Default: {TV_WEIGHT_DEFAULT}."
            ),
        )


def make_requested_diffmap(*, args: argparse.Namespace, mapset: DiffMapSet) -> Map:
    """Compute a difference map, k-weighting as requested"""
    if args.kweight_mode == WeightMode.optimize:
        diffmap, opt_k = max_negentropy_kweighted_difference_map(mapset.derivative, mapset.native)
        log.info("Computing max negentropy diffmap with optimized k-parameter:", value=opt_k)

    elif args.kweight_mode == WeightMode.fixed:
        diffmap = compute_kweighted_difference_map(
            mapset.derivative, mapset.native, k_parameter=args.kweight_parameter
        )
        log.info("Computing diffmap with user-specified k-parameter:", value=args.kweight_parameter)

    elif args.kweight_mode == WeightMode.none:
        diffmap = compute_difference_map(mapset.derivative, mapset.native)
        log.info("Computing unweighted difference map.")

    else:
        raise InvalidWeightModeError(args.kweight_mode)

    return diffmap


def denoise_diffmap(*, args: argparse.Namespace, diffmap: Map) -> tuple[Map, TvDenoiseResult]:
    if args.tv_denoise_mode == WeightMode.optimize:
        log.info(
            "Searching for max-negentropy TV denoising weight",
            min=np.min(TV_LAMBDA_VALUES_TO_SCAN),
            max=np.max(TV_LAMBDA_VALUES_TO_SCAN),
            points_to_test=len(TV_LAMBDA_VALUES_TO_SCAN),
        )
        log.info("This may take some time...")
        final_map, metadata = tv_denoise_difference_map(
            diffmap, full_output=True, weights_to_scan=TV_LAMBDA_VALUES_TO_SCAN
        )
        log.info(
            "Optimal TV weight found, map denoised.",
            best_weight=metadata.optimal_weight,
            initial_negentropy=metadata.initial_negentropy,
            final_negetropy=round(metadata.optimal_negentropy, 3),
        )

    elif args.tv_denoise_mode == WeightMode.fixed:
        log.info("TV denoising with fixed weight", weight=args.tv_weight)
        final_map, metadata = tv_denoise_difference_map(
            diffmap, full_output=True, weights_to_scan=[args.tv_weight]
        )
        log.info(
            "Map TV-denoised with fixed weight.",
            weight=args.tv_weight,
            initial_negentropy=metadata.initial_negentropy,
            final_negetropy=round(metadata.optimal_negentropy, 3),
        )

    elif args.tv_denoise_mode == WeightMode.none:
        final_map = diffmap
        log.info("Requested no denoising. Skipping TV step.")

    else:
        raise InvalidWeightModeError(args.tv_denoise_mode)

    return final_map, metadata


def main(command_line_arguments: list[str] | None = None) -> None:
    parser = TvDiffmapArgParser(
        description="Compute a difference map using native, derivative, and calculated MTZ files."
    )
    args = parser.parse_args(command_line_arguments)
    parser.check_output_filepaths(args)
    mapset = parser.load_difference_maps(args)

    diffmap = make_requested_diffmap(args=args, mapset=mapset)
    final_map, metadata = denoise_diffmap(args=args, diffmap=diffmap)

    log.info("Writing output.", file=args.mtzout)
    final_map.write_mtz(args.mtzout)

    log.info("Writing metadata.", file=args.metadataout)
    metadata.write_csv(args.metadataout)


if __name__ == "__main__":
    main()
