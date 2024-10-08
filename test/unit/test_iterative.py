import numpy as np
import pandas.testing as pdt
import pytest
import reciprocalspaceship as rs

from meteor import iterative
from meteor.utils import assert_phases_allclose


def test_dataseries_l1_norm() -> None:
    series1 = rs.DataSeries([0, 0, 0], index=[0, 1, 2])
    series2 = rs.DataSeries([1, 1, 1], index=[0, 1, 3])
    norm = iterative._dataseries_l1_norm(series1, series2)
    assert norm == 1.0


def test_dataseries_l1_norm_no_overlapping_indices() -> None:
    series1 = rs.DataSeries([0, 0, 0], index=[0, 1, 2])
    series2 = rs.DataSeries([1, 1, 1], index=[3, 4, 5])
    with pytest.raises(RuntimeError):
        iterative._dataseries_l1_norm(series1, series2)


def test_projected_derivative_phase_identical_phases() -> None:
    hkls = [0, 1, 2]
    phases = rs.DataSeries([0.0, 30.0, 60.0], index=hkls)
    amplitudes = rs.DataSeries([1.0, 1.0, 1.0], index=hkls)

    derivative_phases = iterative._projected_derivative_phase(
        difference_amplitudes=amplitudes,
        difference_phases=phases,
        native_amplitudes=amplitudes,
        native_phases=phases,
    )
    assert_phases_allclose(phases.to_numpy(), derivative_phases.to_numpy())


def test_projected_derivative_phase_opposite_phases() -> None:
    hkls = [0, 1, 2]
    native_phases = rs.DataSeries([0.0, 30.0, 60.0], index=hkls)

    # if DF = 0, then derivative and native phase should be the same
    derivative_phases = iterative._projected_derivative_phase(
        difference_amplitudes=rs.DataSeries([0.0, 0.0, 0.0], index=hkls),
        difference_phases=native_phases,
        native_amplitudes=rs.DataSeries([1.0, 1.0, 1.0], index=hkls),
        native_phases=native_phases,
    )
    np.testing.assert_almost_equal(derivative_phases.to_numpy(), native_phases.to_numpy())


def test_iterative_tv(displaced_atom_two_datasets_noise_free: rs.DataSet) -> None:
    result = iterative.iterative_tv_phase_retrieval(displaced_atom_two_datasets_noise_free)
    for label in ["F", "Fh"]:
        pdt.assert_series_equal(
            result[label], displaced_atom_two_datasets_noise_free[label], atol=1e-3
        )
    assert_phases_allclose(
        result["PHIC"], displaced_atom_two_datasets_noise_free["PHIC"], atol=1e-3
    )
    # assert_phases_allclose(
    #     result["PHICh"], displaced_atom_two_datasets_noise_free["PHICh_ground_truth"], atol=1e-3
    # )
