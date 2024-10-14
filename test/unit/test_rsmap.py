import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.rsmap import Map


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


def test_set_uncertainties(noise_free_map: Map) -> None:
    uncertainties = noise_free_map.uncertainties
    assert noise_free_map.has_uncertainties

    noise_free_map.drop(noise_free_map._uncertainty_column, axis=1, inplace=True)
    assert not noise_free_map.has_uncertainties
    with pytest.raises(AttributeError):
        _ = noise_free_map.uncertainties

    noise_free_map.uncertainties = uncertainties
    pd.testing.assert_series_equal(noise_free_map.uncertainties, uncertainties)


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
