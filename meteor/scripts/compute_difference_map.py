import structlog

import numpy as np
from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    max_negentropy_kweighted_difference_map,
)
from meteor.tv import tv_denoise_difference_map

from .common import DiffmapArgParser, load_and_scale_mapset

log = structlog.get_logger()


TV_LAMBDA_VALUES_TO_SCAN = np.linspace(0.0, 0.1, 10)

class TvDiffmapArgParser(DiffmapArgParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "--no-denoising",
            type=bool,
            default=False,
            action="store_true",
            help="Turn off TV denoising and compute a normal isomorphous difference map",
        )


def parse_args():
    parser = TvDiffmapArgParser(
        description="Compute a difference map using native, derivative, and calculated MTZ files."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mapset = load_and_scale_mapset(args)

    if :
        diffmap, opt_k = max_negentropy_kweighted_difference_map(mapset.derivative, mapset.native)
        log.info("Optimal k-parameter determined:", value=opt_k)
    elif args.k_weight_with_fixed_parameter:
        diffmap = compute_kweighted_difference_map(
            mapset.derivative, mapset.native, k_parameter=args.k_weight_with_fixed_parameter
        )
    else:
        diffmap = compute_difference_map(mapset.derivative, mapset.native)

    print("Writing output", file=args.output)
    diffmap.write_mtz(args.output)



    denoised_map, tv_result = tv_denoise_difference_map(
        diffmap, full_output=True, lambda_values_to_scan=TV_LAMBDA_VALUES_TO_SCAN
    )

    log.ingo("Writing output file", optimal_lambda=tv_result.optimal_lambda)
    denoised_map.write_mtz(args.output)

if __name__ == "__main__":
    main()
