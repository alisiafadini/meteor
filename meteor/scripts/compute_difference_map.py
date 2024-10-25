import argparse
from typing import Any

import numpy as np
import structlog

from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    max_negentropy_kweighted_difference_map,
)
from meteor.settings import TV_WEIGHT_DEFAULT
from meteor.tv import tv_denoise_difference_map

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


def parse_args() -> argparse.Namespace:
    parser = TvDiffmapArgParser(
        description="Compute a difference map using native, derivative, and calculated MTZ files."
    )
    return parser.parse_args()


def load_maps() -> DiffMapSet:
    parser = TvDiffmapArgParser()
    return parser.load_difference_maps()


def main() -> None:
    args = parse_args()
    mapset = load_maps()

    # k-weighting
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

    # TV denoising
    if args.tv_denoise_mode == WeightMode.optimize:
        log.info(
            "Searching for max-negentropy TV denoising weight",
            min=np.min(TV_LAMBDA_VALUES_TO_SCAN),
            max=np.max(TV_LAMBDA_VALUES_TO_SCAN),
            points_to_test=len(TV_LAMBDA_VALUES_TO_SCAN),
        )
        log.info("This may take some time...")
        final_map, tv_result = tv_denoise_difference_map(  # TODO: fix result fmt
            diffmap, full_output=True, lambda_values_to_scan=TV_LAMBDA_VALUES_TO_SCAN
        )
        log.info("Optimal TV weight found, map denoised.")  # TODO: print more info

    elif args.tv_denoise_mode == WeightMode.fixed:
        log.info("TV denoising with fixed weight", weight=args.tv_weight)
        final_map, tv_result = tv_denoise_difference_map(  # TODO: fix result fmt
            diffmap, full_output=True, lambda_values_to_scan=[args.tv_weight]
        )
        # TODO: print more info

    elif args.tv_denoise_mode == WeightMode.none:
        log.info("Requested no denoising. Skipping denoising step.")

    else:
        raise InvalidWeightModeError(args.tv_denoise_mode)

    log.info("Writing output.", file=args.output)
    diffmap.write_mtz(args.output)


if __name__ == "__main__":
    main()
