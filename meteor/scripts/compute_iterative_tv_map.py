from __future__ import annotations

from typing import Any

import structlog

from meteor.iterative import iterative_tv_phase_retrieval
from meteor.tv import tv_denoise_difference_map

from .common import DiffmapArgParser, kweight_diffmap_according_to_mode, write_combined_metadata

log = structlog.get_logger()


# TODO: test this
TV_WEIGHTS_TO_SCAN_DEFAULT = [0.01]


class IterativeTvArgParser(DiffmapArgParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_argument(
            "-x",
            "--tv-weights-to-scan",
            nargs="+",
            type=float,
            default=TV_WEIGHTS_TO_SCAN_DEFAULT,
            help=(
                "Choose what TV weights to evaluate at every iteration. Can be a single float."
                f"Default: {TV_WEIGHTS_TO_SCAN_DEFAULT}."
            ),
        )


def main(command_line_arguments: list[str] | None = None) -> None:
    parser = IterativeTvArgParser(
        description=(
            "Compute an difference map, where the phases of the derivative structure are estimated "
            "using the assumption that the resulting map should have a low total variation. Phases "
            "are estimated using a crystallographic analog of the Gerchberg-Saxton algorithm, with "
            "TV denoising as the real-space constraint.\n\n K-weighting can optionally be used to "
            "weight the algorithm input. \n\n In the terminology adopted, this script computes a "
            "`derivative` minus a `native` map, modifying the derivative phases. Native phases,"
            "typically from a model of the `native` data, are computed from a CIF/PDB model you "
            "must provide."
        )
    )
    args = parser.parse_args(command_line_arguments)
    parser.check_output_filepaths(args)
    mapset = parser.load_difference_maps(args)

    # First, find improved derivative phases
    new_derivative_map, it_tv_metadata = iterative_tv_phase_retrieval(
        mapset.derivative,
        mapset.native,
        tv_weights_to_scan=args.tv_weights_to_scan,
    )
    mapset.derivative = new_derivative_map

    diffmap, kparameter_used = kweight_diffmap_according_to_mode(
        kweight_mode=args.kweight_mode, kweight_parameter=args.kweight_parameter, mapset=mapset
    )

    final_map, final_tv_metadata = tv_denoise_difference_map(diffmap, full_output=True)

    log.info("Writing output.", file=str(args.mtzout))
    final_map.write_mtz(args.mtzout)

    log.info("Writing metadata.", file=str(args.metadataout))
    final_tv_metadata.k_parameter_used = kparameter_used
    write_combined_metadata(
        filename=args.metadataout,
        it_tv_metadata=it_tv_metadata,
        final_tv_metadata=final_tv_metadata,
    )


if __name__ == "__main__":
    main()
