import argparse
from dataclasses import dataclass

import structlog
from pathlib import Path

from meteor.rsmap import Map
from meteor.scale import scale_maps
from enum import StrEnum, auto
import re

log = structlog.get_logger()

FLOAT_REGEX = re.compile("^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")


class KweightMode(StrEnum):
    no_weighting = auto()
    fixed_value = auto()
    optimize = auto()


# TODO: better name!
@dataclass
class MapSet:
    native: Map
    derivative: Map
    calculated_native: Map


class DiffmapArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # TODO: add descriptions
        self.add_map_arguments("native", description="...")
        self.add_map_arguments("derivative", description="...")
        self.add_calc_map_arguments("calculated", description="...")

        self.add_argument(
            "-o",
            "--mtzout",
            type=str,
            default="meteor_difference_map.mtz",
            help="Specify output MTZ file name/path.",
        )

        self.add_argument(
            "--kweight",
            type=str,
            default="auto",
            help="Choose the k-weighting behavior. Options: `auto` (choose best negentropy value), `none`, or pass a float for a fixed k-weight.",
        )


    # this will be replaced in a future PR by a more automated parser
    def add_map_arguments(self, map_name: str, *, description: str):
        map_group = self.add_argument_group(map_name, description=description)
        map_group.add_argument("filename", type=Path, required=True)
        map_group.add_argument("--amplitude-label", type=str, default="F", required=True)
        map_group.add_argument("--uncertainty-label", type=str, default="SIGF")
        map_group.add_argument("--phase-label", type=str, default="PHI")
        
    # this will be replaced in a future PR by calculations from a PDB
    def add_calc_map_arguments(self, map_name: str, *, description: str):
        map_group = self.add_argument_group(map_name, description=description)
        map_group.add_argument("filename", type=Path, required=True)
        map_group.add_argument("--amplitude-label", type=str, default="FC", required=True)
        map_group.add_argument("--phase-label", type=str, default="PHIC", required=True)

    @property
    def fixed_kweight_value(self) -> float | None:
        args = self.parse_args()
        regex_group = re.match(FLOAT_REGEX, args.kweight)
        if regex_group is None:
            return None
        return float(regex_group.group(0))

    @property
    def k_weight_mode(self) -> KweightMode:
        args = self.parse_args()
        kweight_arg: str = args.kweight.lower()
        if kweight_arg in ["none", "no", "false"]:
            return KweightMode.no_weighting
        elif kweight_arg in ["auto", "optimize"]:
            return KweightMode.optimize
        elif self.fixed_kweight_value is not None:
            return KweightMode.fixed_value
        msg = f"`{args.kweight}` invalid for --kweight; choose 'auto', 'none', or pass a float"
        raise ValueError(msg)




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
