import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.rsmap import Map


def test_column_names(noise_free_map: Map) -> None:
    # default
    assert noise_free_map.amplitude_column == "F"
    assert noise_free_map.phase_column == "PHI"
    assert noise_free_map.uncertainty_column == "SIGF"
    assert np.all(noise_free_map.columns == ["F", "PHI", "SIGF"])

    # rename
    noise_free_map.amplitude_column = "F2"
    noise_free_map.phase_column = "PHI2"
    noise_free_map.uncertainty_column = "SIGF2"

    # check really renamed
    assert noise_free_map.amplitude_column == "F2"
    assert noise_free_map.phase_column == "PHI2"
    assert noise_free_map.uncertainty_column == "SIGF2"
    assert np.all(noise_free_map.columns == ["F2", "PHI2", "SIGF2"])


def test_setitem(noise_free_map: Map) -> None:
    noise_free_map[noise_free_map.amplitude_column] = noise_free_map.amplitudes
    noise_free_map[noise_free_map.phase_column] = noise_free_map.phases
    noise_free_map[noise_free_map.uncertainty_column] = noise_free_map.uncertainties

    with pytest.raises(KeyError):
        noise_free_map["unallowed_column_name"] = noise_free_map.amplitudes


def test_insert_disabled(noise_free_map: Map) -> None:
    with pytest.raises(NotImplementedError):
        noise_free_map.insert("foo")


def test_set_uncertainties(noise_free_map: Map) -> None:
    assert type(noise_free_map) is Map
    uncertainties = noise_free_map.uncertainties

    assert hasattr(noise_free_map, "_uncertainty_column")
    noise_free_map.drop(noise_free_map.uncertainty_column, axis=1, inplace=True)

    assert type(noise_free_map) is Map, "***"
    delattr(noise_free_map, "_uncertainty_column")
    with pytest.raises(KeyError):
        _ = noise_free_map.uncertainties

    noise_free_map.set_uncertainties(uncertainties)
    pd.testing.assert_series_equal(noise_free_map.uncertainties, uncertainties)


def test_complex(noise_free_map: Map) -> None:
    c_array = noise_free_map.complex
    assert isinstance(c_array, np.ndarray)
    assert np.issubdtype(c_array.dtype, np.complexfloating)


def test_to_structurefactor(noise_free_map: Map) -> None:
    c_dataseries = noise_free_map.to_structurefactor()
    c_array = noise_free_map.complex
    np.testing.assert_almost_equal(c_dataseries.to_numpy(), c_array)


def test_to_ccp4_map(noise_free_map: Map) -> None:
    ccp4_map = noise_free_map.to_ccp4_map(map_sampling=3)
    assert ccp4_map.grid.shape == (30, 30, 30)


def test_from_dataset(noise_free_map: Map) -> None:
    map_as_dataset = rs.DataSet(noise_free_map)
    map2 = Map.from_dataset(
        map_as_dataset,
        amplitude_column=noise_free_map.amplitude_column,
        phase_column=noise_free_map.phase_column,
        uncertainty_column=noise_free_map.uncertainty_column
    )
    pd.testing.assert_frame_equal(noise_free_map, map2)


def from_structurefactor(noise_free_map: Map) -> None:
    # first interface, complex array
    map2 = Map.from_structurefactor(noise_free_map.complex, index=noise_free_map.index)
    pd.testing.assert_frame_equal(noise_free_map, map2)

    # second interface, complex DataSeries
    map2 = Map.from_structurefactor(noise_free_map.to_structurefactor())
    pd.testing.assert_frame_equal(noise_free_map, map2)


def test_from_ccp4_map(ccp4_map: gemmi.Ccp4Map) -> None:
    resolution = 1.0
    rsmap = Map.from_ccp4_map(ccp4_map, high_resolution_limit=resolution)
    assert len(rsmap) > 0
