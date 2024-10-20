import structlog

from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    max_negentropy_kweighted_difference_map,
)

from .common import DiffmapArgParser, load_and_scale_mapset

log = structlog.get_logger()


def parse_args():
    parser = DiffmapArgParser(
        description="Compute a difference map using native, derivative, and calculated MTZ files."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mapset = load_and_scale_mapset(args)

    if args.k_weight_with_parameter_optimization:
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


if __name__ == "__main__":
    main()
