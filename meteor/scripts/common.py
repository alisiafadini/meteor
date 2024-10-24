import argparse
from dataclasses import dataclass

import structlog
from pathlib import Path

from meteor.rsmap import Map
from meteor.scale import scale_maps
from meteor.settings import KWEIGHT_PARAMETER_DEFAULT, COMPUTED_MAP_RESOLUTION_LIMIT
from meteor.sfcalc import structure_to_calculated_map
from meteor.io import find_observed_amplitude_label, find_observed_uncertainty_label

from enum import StrEnum, auto
from typing import Any
import re

import reciprocalspaceship as rs

log = structlog.get_logger()

INFER = "infer"
PHASE_COLUMN_NAME = "PHI"
DEFAULT_OUTPUT_MTZ = Path("meteor_difference_map.mtz")
FLOAT_REGEX = re.compile("^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")


class KweightMode(StrEnum):
    optimize = auto()
    fixed = auto()
    none = auto()


class DiffmapArgParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # TODO: add descriptions
        derivative_group = self.add_argument_group("derivative", description="...")
        derivative_group.add_argument("derivative_mtz", type=Path, required=True)
        derivative_group.add_argument(
            "-da",
            "--derivative-amplitude-label",
            type=str,
            default=INFER,
            help="specify the MTZ column for the amplitudes; will try to guess if not provided",
        )
        derivative_group.add_argument(
            "-du",
            "--derivative-uncertainty-label",
            type=str,
            default=INFER,
            help="specify the MTZ column for the uncertainties; will try to guess if not provided",
        )

        native_group = self.add_argument_group("native", description="...")
        native_group.add_argument("native_mtz", type=Path, required=True)
        native_group.add_argument(
            "-na",
            "--native-amplitude-label",
            type=str,
            default=INFER,
            help="specify the MTZ column for the amplitudes; will try to guess if not provided",
        )
        native_group.add_argument(
            "-nu",
            "--native-uncertainty-label",
            type=str,
            default=INFER,
            help="specify the MTZ column for the uncertainties; will try to guess if not provided",
        )

        self.add_argument(
            "-p",
            "--pdb",
            type=Path,
            required=True,
            help="Specify PDB file name/path, model should correspond to the native MTZ.",
        )

        self.add_argument(
            "-o",
            "--mtzout",
            type=str,
            default=DEFAULT_OUTPUT_MTZ,
            help=f"Specify output MTZ file name/path. Default: {DEFAULT_OUTPUT_MTZ}.",
        )

        self.add_argument(
            "-k",
            "--kweight-mode",
            type=str,
            default=KweightMode.optimize,
            choices=KweightMode,
            help="Choose the k-weighting behavior.",
        )

        self.add_argument(
            "-w",
            "--kweight-parameter",
            type=float,
            default=KWEIGHT_PARAMETER_DEFAULT,
            help=f"If --kweight-mode == {KweightMode.fixed}, set the kweight-parameter to this value. Default: {KWEIGHT_PARAMETER_DEFAULT}.",
        )


@dataclass
class DiffMapSet:
    native: Map
    derivative: Map
    calculated: Map

    @staticmethod
    def _construct_map(
        *,
        name: str,
        mtz_file: Path,
        calculated_map_phases: rs.DataSeries,
        amplitude_label: str,
        uncertainty_label: str,
    ) -> Map:
        mtz = rs.read_mtz(mtz_file)
        mtz[PHASE_COLUMN_NAME] = calculated_map_phases

        found_amplitude_column = (
            find_observed_amplitude_label(mtz.columns)
            if amplitude_label is INFER
            else amplitude_label
        )
        found_uncertainty_column = (
            find_observed_uncertainty_label(mtz.columns)
            if uncertainty_label is INFER
            else uncertainty_label
        )

        log.info(
            "Loading",
            map=name,
            file=mtz_file,
            amplitudes=found_amplitude_column,
            uncertainties=found_uncertainty_column,
        )

        return Map(
            mtz,
            amplitude_column=found_amplitude_column,
            phase_column=PHASE_COLUMN_NAME,
            uncertainty_column=found_uncertainty_column,
        )

    def scale(self) -> None:
        # note: FC do not have uncertainties
        # TODO: enable weighting with single uncertainties
        self.native = scale_maps(
            reference_map=self.calculated,
            map_to_scale=self.native,
            weight_using_uncertainties=True,
        )
        log.info("scaling: native map --> calculated native")

        self.derivative = scale_maps(
            reference_map=self.calculated,
            map_to_scale=self.derivative,
            weight_using_uncertainties=True,
        )
        log.info("scaling: derivative map --> calculated native")

    @classmethod
    def from_arguments(cls, args: argparse.Namespace) -> DiffMapSet:
        calculated_map = structure_to_calculated_map(
            args.pdb, high_resolution_limit=COMPUTED_MAP_RESOLUTION_LIMIT
        )
        log.info("Loading PDB & computing FC/PHIC", file=args.pdb)

        derivative_map = cls._construct_map(
            name="derivative",
            mtz_file=args.derivative_mtz,
            calculated_map_phases=calculated_map.phases,
            amplitude_label=args.derivative_amplitude_label,
            uncertainty_label=args.derivative_uncertainty_label,
        )

        native_map = cls._construct_map(
            name="native",
            mtz_file=args.native_mtz,
            calculated_map_phases=calculated_map.phases,
            amplitude_label=args.native_amplitude_label,
            uncertainty_label=args.native_uncertainty_label,
        )

        mapset = cls(
            native=native_map,
            derivative=derivative_map,
            calculated=calculated_map,
        )

        mapset.scale()
        return mapset
