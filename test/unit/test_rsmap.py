import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.rsmap import Map


def test_copy(noise_free_map: Map) -> None:
    copy_map = noise_free_map.copy()
    assert isinstance(copy_map, Map)
    pd.testing.assert_frame_equal(copy_map, noise_free_map)


def test_setitem(noise_free_map: Map, noisy_map: Map) -> None:
    noisy_map.amplitudes = noise_free_map.amplitudes
    noisy_map.phases = noise_free_map.phases
    noisy_map.uncertainties = noise_free_map.uncertainties


def test_unallowed_setitem(noise_free_map: Map) -> None:
    with pytest.raises(KeyError):
        noise_free_map["unallowed_column_name"] = noise_free_map.amplitudes


def test_insert_disabled(noise_free_map: Map) -> None:
    with pytest.raises(NotImplementedError):
        noise_free_map.insert("foo")


def test_set_uncertainties() -> None:
    test_map =  Map.from_dict(
        {
            "F": rs.DataSeries([2.0, 3.0, 4.0]).astype(rs.StructureFactorAmplitudeDtype()), 
            "PHI": rs.DataSeries([0.0, 0.0, 0.0]).astype(rs.PhaseDtype())
        }
    )

    assert not test_map.has_uncertainties
    with pytest.raises(AttributeError):
        _ = test_map.uncertainties

    test_map.uncertainties = rs.DataSeries([1.0, 1.0, 1.0]).astype(rs.StandardDeviationDtype())
    assert test_map.has_uncertainties
    assert len(test_map.uncertainties) == 3


def test_complex_amplitudes(noise_free_map: Map) -> None:
    c_array = noise_free_map.complex_amplitudes
    assert isinstance(c_array, np.ndarray)
    assert np.issubdtype(c_array.dtype, np.complexfloating)


def test_to_structurefactor(noise_free_map: Map) -> None:
    c_dataseries = noise_free_map.to_structurefactor()
    c_array = noise_free_map.complex_amplitudes
    np.testing.assert_almost_equal(c_dataseries.to_numpy(), c_array)


def test_to_ccp4_map(noise_free_map: Map) -> None:
    ccp4_map = noise_free_map.to_ccp4_map(map_sampling=3)
    assert ccp4_map.grid.shape == (30, 30, 30)


def test_from_dataset(noise_free_map: Map) -> None:
    map_as_dataset = rs.DataSet(noise_free_map)
    map2 = Map(
        map_as_dataset,
        amplitude_column=noise_free_map._amplitude_column,
        phase_column=noise_free_map._phase_column,
        uncertainty_column=noise_free_map._uncertainty_column,
    )
    pd.testing.assert_frame_equal(noise_free_map, map2)


def from_structurefactor(noise_free_map: Map) -> None:
    # first interface, complex array
    map2 = Map.from_structurefactor(noise_free_map.complex_amplitudes, index=noise_free_map.index)
    pd.testing.assert_frame_equal(noise_free_map, map2)

    # second interface, complex DataSeries
    map2 = Map.from_structurefactor(noise_free_map.to_structurefactor())
    pd.testing.assert_frame_equal(noise_free_map, map2)


def test_from_ccp4_map(ccp4_map: gemmi.Ccp4Map) -> None:
    resolution = 1.0
    rsmap = Map.from_ccp4_map(ccp4_map, high_resolution_limit=resolution)
    assert len(rsmap) > 0
