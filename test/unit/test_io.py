from __future__ import annotations

from typing import Callable

import pytest

from meteor import io

FIND_LABEL_FUNC_TYPE = Callable[[list[str]], str]


OBSERVED_INTENSITY_CASES = [
    ([], "raise"),
    (["F"], "raise"),
    (["I", "F"], "I"),
    (["IMEAN", "F"], "IMEAN"),
    (["I-obs", "F"], "I-obs"),
    (["I", "IMEAN"], "raise"),
]


OBSERVED_AMPLITUDE_CASES = [
    ([], "raise"),
    (["IMEAN"], "raise"),
    (["I", "F"], "F"),
    (["IMEAN", "FP"], "FP"),
    (["I-obs", "FPH0"], "FPH0"),
    (["I-obs", "FPH5"], "FPH5"),
    (["I-obs", "FPH51"], "raise"),
    (["FP", "FPH1"], "raise"),
]


OBSERVED_UNCERTAINTY_CASES = [
    ([], "raise"),
    (["F"], "raise"),
    (["SIGF", "F"], "SIGF"),
    (["SIGFP", "F"], "SIGFP"),
    (["I-obs", "SIGFPH0"], "SIGFPH0"),
    (["I-obs", "SIGFPH1"], "SIGFPH1"),
    (["I-obs", "SIGFPH10"], "raise"),
    (["SIGFPH1", "SIGFPH2"], "raise"),
]


COMPUTED_AMPLITUDE_CASES = [
    ([], "raise"),
    (["F"], "raise"),
    (["I", "F"], "raise"),
    (["FC", "F"], "FC"),
    (["FC", "FC"], "raise"),
]


COMPUTED_PHASE_CASES = [
    ([], "raise"),
    (["F"], "raise"),
    (["PHIC", "F"], "PHIC"),
    (["PHIC", "PHIC"], "raise"),
]


def test_infer_mtz_label() -> None:
    to_search = ["FOO", "BAR", "BAZ"]
    assert io._infer_mtz_label(to_search, ["FOO"]) == "FOO"
    assert io._infer_mtz_label(to_search, ["BAR"]) == "BAR"
    with pytest.raises(io.AmbiguousMtzLabelError):
        _ = io._infer_mtz_label(to_search, [])
    with pytest.raises(io.AmbiguousMtzLabelError):
        _ = io._infer_mtz_label(to_search, ["FOO", "BAR"])


def validate_find_label_result(
    function: FIND_LABEL_FUNC_TYPE, labels: list[str], expected_result: str
) -> None:
    if expected_result == "raise":
        with pytest.raises(io.AmbiguousMtzLabelError):
            _ = function(labels)
    else:
        assert function(labels) == expected_result


@pytest.mark.parametrize(("labels", "expected_result"), OBSERVED_INTENSITY_CASES)
def test_find_observed_intensity_label(labels: list[str], expected_result: str) -> None:
    validate_find_label_result(io.find_observed_intensity_label, labels, expected_result)


@pytest.mark.parametrize(("labels", "expected_result"), OBSERVED_AMPLITUDE_CASES)
def test_find_observed_amplitude_label(labels: list[str], expected_result: str) -> None:
    validate_find_label_result(io.find_observed_amplitude_label, labels, expected_result)


@pytest.mark.parametrize(("labels", "expected_result"), OBSERVED_UNCERTAINTY_CASES)
def test_find_observed_uncertainty_label(labels: list[str], expected_result: str) -> None:
    validate_find_label_result(io.find_observed_uncertainty_label, labels, expected_result)


@pytest.mark.parametrize(("labels", "expected_result"), COMPUTED_AMPLITUDE_CASES)
def test_find_computed_amplitude_label(labels: list[str], expected_result: str) -> None:
    validate_find_label_result(io.find_computed_amplitude_label, labels, expected_result)


@pytest.mark.parametrize(("labels", "expected_result"), COMPUTED_PHASE_CASES)
def test_find_computed_phase_label(labels: list[str], expected_result: str) -> None:
    validate_find_label_result(io.find_computed_phase_label, labels, expected_result)