import numpy as np
import structlog

from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    max_negentropy_kweighted_difference_map,
)
from meteor.tv import tv_denoise_difference_map

from .common import DiffmapArgParser, load_and_scale_mapset

log = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = DiffmapArgParser(
        description="Compute a difference map using native, derivative, and calculated MTZ files."
    )
    # TODO: add type of lambda screen input

    return parser.parse_args()


def main():
    args = parse_args()
    mapset = load_and_scale_mapset(args)

    # TODO: eliminate this copypasta as well
    # question: do we want to enforce k-weighting? if not, I would say maybe we just combine
    # the two scripts and add a single --no-tv-denoising flag or something...
    if args.k_weight_with_parameter_optimization:
        diffmap, opt_k = max_negentropy_kweighted_difference_map(mapset.derivative, mapset.native)
        log.info("Optimal k-parameter determined:", value=opt_k)
    elif args.k_weight_with_fixed_parameter:
        diffmap = compute_kweighted_difference_map(
            mapset.derivative, mapset.native, k_parameter=args.k_weight_with_fixed_parameter
        )
    else:
        diffmap = compute_difference_map(mapset.derivative, mapset.native)

    # Now TV denoise
    # TODO: hard coded linspace
    denoised_map, tv_result = tv_denoise_difference_map(
        diffmap, full_output=True, lambda_values_to_scan=np.linspace(1e-8, 0.05, 10)
    )

    log.ingo("Writing output file", optimal_lambda=tv_result.optimal_lambda)
    denoised_map.write_mtz(args.output)


if __name__ == "__main__":
    main()
