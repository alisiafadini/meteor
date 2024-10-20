import argparse
from dataclasses import dataclass

import structlog
from pathlib import Path

from meteor.rsmap import Map
from meteor.scale import scale_maps

log = structlog.get_logger()


# TODO: better name!
@dataclass
class MapSet:
    native: Map
    derivative: Map
    calculated_native: Map


class DiffmapArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.add_argument(
            "--native_mtz",
            nargs=4,
            metavar=("filename", "amplitude_label", "uncertainty_label", "phase_label"),
            required=True,
            help=("Native MTZ file and associated amplitude, uncertainty labels, and phase label."),
        )

        self.add_argument(
            "--derivative_mtz",
            nargs=4,
            metavar=("filename", "amplitude_label", "uncertainty_label", "phase_label"),
            required=True,
            help=(
                "Derivative MTZ file and associated amplitude, uncertainty labels, and phase label."
            ),
        )

        self.add_argument(
            "--calc_native_mtz",
            nargs=3,
            metavar=("filename", "calc_amplitude_label", "calc_phase_label"),
            required=True,
            help=(
                "Calculated native MTZ file and associated calculated amplitude and phase labels."
            ),
        )

        self.add_argument(
            "--output",
            type=str,
            default="meteor_difference_map.mtz",
            help="Output file name",
        )

        self.add_argument(
            "--use_uncertainties_to_scale",
            type=bool,
            default=True,
            help="Use uncertainties to scale (default: True)",
        )

        k_weight_group = self.add_mutually_exclusive_group()

        k_weight_group.add_argument(
            "--k_weight_with_fixed_parameter",
            type=float,
            default=None,
            help="Use k-weighting with a fixed parameter (float between 0 and 1.0)",
        )

        k_weight_group.add_argument(
            "--k_weight_with_parameter_optimization",
            action="store_true",  # This will set the flag to True when specified
            help="Use k-weighting with parameter optimization (default: False)",
        )

    def add_map_arguments(self, map_name: str, *, description: str):
        map_group = self.add_argument_group(map_name, description=description)
        self.add_argument("filename", type=Path, required=True)
        self.add_argument("--amplitude-label", type=str, default="F")



def _log_map_read(map_name: str, args_obj) -> None:
    phases = args_obj[2] if len(args_obj) == 3 else None
    log.info("Read map", name=map_name, amps=args_obj[1], stds=args_obj[2], phases=phases)


def load_and_scale_mapset(args: argparse.Namespace) -> MapSet:
    # Create the native map from the native MTZ file
    native_map = Map.read_mtz_file(
        args.native_mtz[0],
        amplitude_column=args.native_mtz[1],
        uncertainty_column=args.native_mtz[2],
        phase_column=args.native_mtz[3],
    )
    _log_map_read("native", args.native_mtz)

    # Create the derivative map from the derivative MTZ file
    derivative_map = Map.read_mtz_file(
        args.derivative_mtz[0],
        amplitude_column=args.derivative_mtz[1],
        uncertainty_column=args.derivative_mtz[2],
        phase_column=args.derivative_mtz[3],
    )
    _log_map_read("derivative", args.derivative_mtz)

    # Create the calculated native map from the calculated native MTZ file
    calc_native_map = Map.read_mtz_file(
        args.calc_native_mtz[0],
        amplitude_column=args.calc_native_mtz[1],
        phase_column=args.calc_native_mtz[2],
    )
    _log_map_read("calculated native", args.calc_native_mtz)

    # Scale both to common map calculated from native model
    native_map_scaled = scale_maps(
        reference_map=calc_native_map,
        map_to_scale=native_map,
        weight_using_uncertainties=False,
    )
    log.info("scaling: native --> calculated native")
    derivative_map_scaled = scale_maps(
        reference_map=calc_native_map,
        map_to_scale=derivative_map,
        weight_using_uncertainties=False,  # FC do not have uncertainties
    )
    log.info("scaling: derivative --> calculated native")

    return MapSet(
        native=native_map_scaled,
        derivative=derivative_map_scaled,
        calculated_native=calc_native_map,
    )
