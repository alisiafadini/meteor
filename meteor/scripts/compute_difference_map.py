from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    max_negentropy_kweighted_difference_map,
)
from meteor.rsmap import Map
from meteor.settings import MAP_SAMPLING, TV_WEIGHT_DEFAULT
from meteor.tv import TvDenoiseResult, tv_denoise_difference_map
from meteor.validate import negentropy

from .common import DiffmapArgParser, DiffMapSet, InvalidWeightModeError, WeightMode

log = structlog.get_logger()


# TO OPTMIZE
TV_WEIGHTS_TO_SCAN = np.linspace(0.0, 0.1, 101)


class TvDiffmapArgParser(DiffmapArgParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_argument(
            "-tv",
            "--tv-denoise-mode",
            type=WeightMode,
            default=WeightMode.optimize,
            choices=list(WeightMode),
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


def make_requested_diffmap(
    *, mapset: DiffMapSet, kweight_mode: WeightMode, kweight_parameter: float | None = None
) -> Map:
    if kweight_mode == WeightMode.optimize:
        diffmap, opt_k = max_negentropy_kweighted_difference_map(mapset.derivative, mapset.native)
        log.info("Computing max negentropy diffmap with optimized k-parameter:", value=opt_k)

    elif kweight_mode == WeightMode.fixed:
        if not isinstance(kweight_parameter, float):
            msg = f"`kweight_parameter` is type `{type(kweight_parameter)}`, must be `float`"
            raise TypeError(msg)
        diffmap = compute_kweighted_difference_map(
            mapset.derivative, mapset.native, k_parameter=kweight_parameter
        )
        log.info("Computing diffmap with user-specified k-parameter:", value=kweight_parameter)

    elif kweight_mode == WeightMode.none:
        diffmap = compute_difference_map(mapset.derivative, mapset.native)
        log.info("Computing unweighted difference map.")

    else:
        raise InvalidWeightModeError(kweight_mode)

    return diffmap


def denoise_diffmap(
    *,
    diffmap: Map,
    tv_denoise_mode: WeightMode,
    tv_weight: float | None = None,
) -> tuple[Map, TvDenoiseResult]:
    if tv_denoise_mode == WeightMode.optimize:
        log.info(
            "Searching for max-negentropy TV denoising weight",
            min=np.min(TV_WEIGHTS_TO_SCAN),
            max=np.max(TV_WEIGHTS_TO_SCAN),
            points_to_test=len(TV_WEIGHTS_TO_SCAN),
        )
        log.info("This may take some time...")

        final_map, metadata = tv_denoise_difference_map(
            diffmap, full_output=True, weights_to_scan=TV_WEIGHTS_TO_SCAN
        )

        log.info(
            "Optimal TV weight found, map denoised.",
            best_weight=metadata.optimal_weight,
            initial_negentropy=metadata.initial_negentropy,
            final_negetropy=round(metadata.optimal_negentropy, 3),
        )

    elif tv_denoise_mode == WeightMode.fixed:
        if not isinstance(tv_weight, float):
            msg = f"`tv_weight` is type `{type(tv_weight)}`, must be `float`"
            raise TypeError(msg)

        log.info("TV denoising with fixed weight", weight=tv_weight)
        final_map, metadata = tv_denoise_difference_map(
            diffmap, full_output=True, weights_to_scan=[tv_weight]
        )

        log.info(
            "Map TV-denoised with fixed weight.",
            weight=tv_weight,
            initial_negentropy=metadata.initial_negentropy,
            final_negetropy=round(metadata.optimal_negentropy, 3),
        )

    elif tv_denoise_mode == WeightMode.none:
        final_map = diffmap

        realspace_map = final_map.to_ccp4_map(map_sampling=MAP_SAMPLING)
        map_negetropy = negentropy(np.array(realspace_map.grid))
        metadata = TvDenoiseResult(
            initial_negentropy=map_negetropy,
            optimal_negentropy=map_negetropy,
            optimal_weight=0.0,
            map_sampling_used_for_tv=MAP_SAMPLING,
            weights_scanned=[0.0],
            negentropy_at_weights=[map_negetropy],
        )

        log.info("Requested no denoising. Skipping TV step.")

    else:
        raise InvalidWeightModeError(tv_denoise_mode)

    return final_map, metadata


def main(command_line_arguments: list[str] | None = None) -> None:
    parser = TvDiffmapArgParser(
        description="Compute a difference map using native, derivative, and calculated MTZ files."
    )
    args = parser.parse_args(command_line_arguments)
    parser.check_output_filepaths(args)
    mapset = parser.load_difference_maps(args)

    diffmap = make_requested_diffmap(
        kweight_mode=args.kweight_mode, kweight_parameter=args.kweight_parameter, mapset=mapset
    )
    final_map, metadata = denoise_diffmap(
        tv_denoise_mode=args.tv_denoise_mode, tv_weight=args.tv_weight, diffmap=diffmap
    )

    log.info("Writing output.", file=args.mtzout)
    final_map.write_mtz(args.mtzout)

    log.info("Writing metadata.", file=args.metadataout)
    metadata.write_csv(args.metadataout)


if __name__ == "__main__":
    main()
