import argparse
import re
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

import reciprocalspaceship as rs
import structlog

from meteor.io import find_observed_amplitude_column, find_observed_uncertainty_column
from meteor.rsmap import Map
from meteor.scale import scale_maps
from meteor.settings import COMPUTED_MAP_RESOLUTION_LIMIT, KWEIGHT_PARAMETER_DEFAULT
from meteor.sfcalc import pdb_to_calculated_map

log = structlog.get_logger()

INFER_COLUMN_NAME: str = "infer"
PHASE_COLUMN_NAME: str = "PHI"
DEFAULT_OUTPUT_MTZ: Path = Path("meteor_difference_map.mtz")
DEFAULT_OUTPUT_METADATA_FILE: Path = Path("meteor_metadata.csv")
FLOAT_REGEX: re.Pattern = re.compile(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")


class InvalidWeightModeError(ValueError): ...


class WeightMode(StrEnum):
    optimize = auto()
    fixed = auto()
    none = auto()


@dataclass
class DiffMapSet:
    native: Map
    derivative: Map
    calculated: Map

    def scale(self, *, weight_using_uncertainties: bool = True) -> None:
        self.native = scale_maps(
            reference_map=self.calculated,
            map_to_scale=self.native,
            weight_using_uncertainties=weight_using_uncertainties,
        )
        log.info(
            "scaling: native --> calculated",
            weight_using_uncertainties=weight_using_uncertainties,
        )

        self.derivative = scale_maps(
            reference_map=self.calculated,
            map_to_scale=self.derivative,
            weight_using_uncertainties=weight_using_uncertainties,
        )
        log.info(
            "scaling: derivative --> calculated",
            weight_using_uncertainties=weight_using_uncertainties,
        )


class DiffmapArgParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        derivative_group = self.add_argument_group(
            "derivative",
            description=(
                "The 'derivative' diffraction data, typically: light-triggered, ligand-bound, etc. "
                "We compute derivative-minus-native maps."
            ),
        )
        derivative_group.add_argument("derivative_mtz", type=Path)
        derivative_group.add_argument(
            "-da",
            "--derivative-amplitude-column",
            type=str,
            default=INFER_COLUMN_NAME,
            help="specify the MTZ column for the amplitudes; will try to guess if not provided",
        )
        derivative_group.add_argument(
            "-du",
            "--derivative-uncertainty-column",
            type=str,
            default=INFER_COLUMN_NAME,
            help="specify the MTZ column for the uncertainties; will try to guess if not provided",
        )

        native_group = self.add_argument_group(
            "native",
            description=(
                "The 'native' diffraction data, typically: dark, apo, etc. We compute derivative-"
                "minus-native maps. The single set of known phases are typically assumed to "
                "correspond to the native dataset."
            ),
        )
        native_group.add_argument("native_mtz", type=Path)
        native_group.add_argument(
            "-na",
            "--native-amplitude-column",
            type=str,
            default=INFER_COLUMN_NAME,
            help="specify the MTZ column for the amplitudes; will try to guess if not provided",
        )
        native_group.add_argument(
            "-nu",
            "--native-uncertainty-column",
            type=str,
            default=INFER_COLUMN_NAME,
            help="specify the MTZ column for the uncertainties; will try to guess if not provided",
        )

        self.add_argument(
            "-p",
            "--pdb",
            type=Path,
            required=True,
            help="Specify PDB file path, model should correspond to the native MTZ.",
        )

        self.add_argument(
            "-o",
            "--mtzout",
            type=Path,
            default=DEFAULT_OUTPUT_MTZ,
            help=f"Specify output MTZ file path. Default: {DEFAULT_OUTPUT_MTZ}.",
        )

        self.add_argument(
            "-m",
            "--metadataout",
            type=Path,
            default=DEFAULT_OUTPUT_METADATA_FILE,
            help=f"Specify output metadata file path. Default: {DEFAULT_OUTPUT_METADATA_FILE}.",
        )

        self.add_argument(
            "-k",
            "--kweight-mode",
            type=WeightMode,
            default=WeightMode.optimize,
            choices=list(WeightMode),
            help="Choose the k-weighting behavior.",
        )

        self.add_argument(
            "-w",
            "--kweight-parameter",
            type=float,
            default=KWEIGHT_PARAMETER_DEFAULT,
            help=(
                f"If `--kweight-mode {WeightMode.fixed}`, set the kweight-parameter to this value. "
                f"Default: {KWEIGHT_PARAMETER_DEFAULT}."
            ),
        )

    @staticmethod
    def check_output_filepaths(args: argparse.Namespace) -> None:
        for filename in [args.mtzout, args.metadataout]:
            if filename.exists():
                msg = f"file: {filename} already exists, refusing to overwrite"
                raise OSError(msg)

    @staticmethod
    def _construct_map(
        *,
        name: str,
        mtz_file: Path,
        calculated_map_phases: rs.DataSeries,
        amplitude_column: str,
        uncertainty_column: str,
    ) -> Map:
        log.info(
            "Reading structure factors...",
            file=str(mtz_file),
            map=name,
        )

        mtz = rs.read_mtz(str(mtz_file))
        if PHASE_COLUMN_NAME in mtz.columns:
            log.warning(
                "phase column already in MTZ; overwriting with computed data",
                file=str(mtz_file),
                column=PHASE_COLUMN_NAME,
            )
        mtz[PHASE_COLUMN_NAME] = calculated_map_phases

        found_amplitude_column = (
            find_observed_amplitude_column(mtz.columns)
            if amplitude_column is INFER_COLUMN_NAME
            else amplitude_column
        )
        log.info("  amplitudes", sought=amplitude_column, found=found_amplitude_column)

        found_uncertainty_column = (
            find_observed_uncertainty_column(mtz.columns)
            if uncertainty_column is INFER_COLUMN_NAME
            else uncertainty_column
        )
        log.info("  uncertainties", sought=uncertainty_column, found=found_uncertainty_column)

        return Map(
            mtz,
            amplitude_column=found_amplitude_column,
            phase_column=PHASE_COLUMN_NAME,
            uncertainty_column=found_uncertainty_column,
        )

    @staticmethod
    def load_difference_maps(args: argparse.Namespace) -> DiffMapSet:
        # note: method accepts `args`, in case the passed arguments are mutable

        log.info("Loading PDB & computing FC/PHIC", file=str(args.pdb))
        calculated_map = pdb_to_calculated_map(
            args.pdb, high_resolution_limit=COMPUTED_MAP_RESOLUTION_LIMIT
        )

        derivative_map = DiffmapArgParser._construct_map(
            name="derivative",
            mtz_file=args.derivative_mtz,
            calculated_map_phases=calculated_map.phases,
            amplitude_column=args.derivative_amplitude_column,
            uncertainty_column=args.derivative_uncertainty_column,
        )

        native_map = DiffmapArgParser._construct_map(
            name="native",
            mtz_file=args.native_mtz,
            calculated_map_phases=calculated_map.phases,
            amplitude_column=args.native_amplitude_column,
            uncertainty_column=args.native_uncertainty_column,
        )

        mapset = DiffMapSet(
            native=native_map,
            derivative=derivative_map,
            calculated=calculated_map,
        )

        mapset.scale()
        return mapset
