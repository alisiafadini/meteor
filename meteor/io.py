"""https://www.ccp4.ac.uk/html/mtzformat.html
https://www.globalphasing.com/buster/wiki/index.cgi?MTZcolumns
"""

from __future__ import annotations

import re
from typing import Final

# TODO: scour for PHENIX style, add

OBSERVED_INTENSITY_LABELS: Final[list[str]] = [
    "I",  # generic
    "IMEAN",  # CCP4
    "I-obs",  # phenix
]

OBSERVED_AMPLITUDE_LABELS: Final[list[str]] = [
    "F",  # generic
    "FP",  # CCP4 & GLPh native
    r"FPH\d",  # CCP4 derivative
    "F-obs",  # phenix
]

OBSERVED_UNCERTAINTY_LABELS: Final[list[str]] = [
    "SIGF",  # generic
    "SIGFP",  # CCP4 & GLPh native
    r"SIGFPH\d",  # CCP4
]

COMPUTED_AMPLITUDE_LABELS: Final[list[str]] = ["FC"]

COMPUTED_PHASE_LABELS: Final[list[str]] = ["PHIC"]


class AmbiguousMtzLabelError(ValueError): ...


def _infer_mtz_label(labels_to_search: list[str], labels_to_look_for: list[str]) -> str:
    # the next line consumes ["FOO", "BAR", "BAZ"] and produces regex strings like "^(FOO|BAR|BAZ)$"
    regex = re.compile(f"^({'|'.join(labels_to_look_for)})$")
    matches = [regex.match(label) for label in labels_to_search if regex.match(label) is not None]

    if len(matches) == 0:
        msg = "cannot infer MTZ column name; "
        msg += f"cannot find any of {labels_to_look_for} in {labels_to_search}"
        raise AmbiguousMtzLabelError(msg)
    if len(matches) > 1:
        msg = "cannot infer MTZ column name; "
        msg += f">1 instance of {labels_to_look_for} in {labels_to_search}"
        raise AmbiguousMtzLabelError(msg)

    [match] = matches
    if match is None:
        msg = "`None` not filtered during regex matching"
        raise RuntimeError(msg)

    return match.group(0)


def find_observed_intensity_label(mtz_column_labels: list[str]) -> str:
    return _infer_mtz_label(mtz_column_labels, OBSERVED_INTENSITY_LABELS)


def find_observed_amplitude_label(mtz_column_labels: list[str]) -> str:
    return _infer_mtz_label(mtz_column_labels, OBSERVED_AMPLITUDE_LABELS)


def find_observed_uncertainty_label(mtz_column_labels: list[str]) -> str:
    return _infer_mtz_label(mtz_column_labels, OBSERVED_UNCERTAINTY_LABELS)


def find_computed_amplitude_label(mtz_column_labels: list[str]) -> str:
    return _infer_mtz_label(mtz_column_labels, COMPUTED_AMPLITUDE_LABELS)


def find_computed_phase_label(mtz_column_labels: list[str]) -> str:
    return _infer_mtz_label(mtz_column_labels, COMPUTED_PHASE_LABELS)
