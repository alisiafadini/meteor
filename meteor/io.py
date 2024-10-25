"""https://www.ccp4.ac.uk/html/mtzformat.html
https://www.globalphasing.com/buster/wiki/index.cgi?MTZcolumns
"""

from __future__ import annotations

import re
from typing import Final

OBSERVED_INTENSITY_columnS: Final[list[str]] = [
    "I",  # generic
    "IMEAN",  # CCP4
    "I-obs",  # phenix
]

OBSERVED_AMPLITUDE_columnS: Final[list[str]] = [
    "F",  # generic
    "FP",  # CCP4 & GLPh native
    r"FPH\d",  # CCP4 derivative
    "F-obs",  # phenix
]

OBSERVED_UNCERTAINTY_columnS: Final[list[str]] = [
    "SIGF",  # generic
    "SIGFP",  # CCP4 & GLPh native
    r"SIGFPH\d",  # CCP4
]

COMPUTED_AMPLITUDE_columnS: Final[list[str]] = ["FC"]

COMPUTED_PHASE_columnS: Final[list[str]] = ["PHIC"]


class AmbiguousMtzcolumnError(ValueError): ...


def _infer_mtz_column(columns_to_search: list[str], columns_to_look_for: list[str]) -> str:
    # the next line consumes ["FOO", "BAR", "BAZ"] and produces regex strings like "^(FOO|BAR|BAZ)$"
    regex = re.compile(f"^({'|'.join(columns_to_look_for)})$")
    matches = [regex.match(column) for column in columns_to_search if regex.match(column) is not None]

    if len(matches) == 0:
        msg = "cannot infer MTZ column name; "
        msg += f"cannot find any of {columns_to_look_for} in {columns_to_search}"
        raise AmbiguousMtzcolumnError(msg)
    if len(matches) > 1:
        msg = "cannot infer MTZ column name; "
        msg += f">1 instance of {columns_to_look_for} in {columns_to_search}"
        raise AmbiguousMtzcolumnError(msg)

    [match] = matches
    if match is None:
        msg = "`None` not filtered during regex matching"
        raise RuntimeError(msg)

    return match.group(0)


def find_observed_intensity_column(mtz_column_columns: list[str]) -> str:
    return _infer_mtz_column(mtz_column_columns, OBSERVED_INTENSITY_columnS)


def find_observed_amplitude_column(mtz_column_columns: list[str]) -> str:
    return _infer_mtz_column(mtz_column_columns, OBSERVED_AMPLITUDE_columnS)


def find_observed_uncertainty_column(mtz_column_columns: list[str]) -> str:
    return _infer_mtz_column(mtz_column_columns, OBSERVED_UNCERTAINTY_columnS)


def find_computed_amplitude_column(mtz_column_columns: list[str]) -> str:
    return _infer_mtz_column(mtz_column_columns, COMPUTED_AMPLITUDE_columnS)


def find_computed_phase_column(mtz_column_columns: list[str]) -> str:
    return _infer_mtz_column(mtz_column_columns, COMPUTED_PHASE_columnS)
