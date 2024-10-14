

import pytest
import gemmi
import numpy as np
import pandas as pd
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
    uncertainties = noise_free_map.uncertainties
    noise_free_map.drop(noise_free_map.uncertainty_column, axis=1, inplace=True)
    
    delattr(noise_free_map, "_uncertainty_column")
    with pytest.raises(KeyError):
        noise_free_map.uncertainties

    noise_free_map.set_uncertainties(uncertainties)
    pd.testing.assert_series_equal(noise_free_map.uncertainties, uncertainties)


def test_complex(noise_free_map: Map) -> None:
    ...


def test_to_structurefactor(noise_free_map: Map) -> None:
    ...


def test_to_gemmi(noise_free_map: Map) -> None:
    ...


def test_from_dataset() -> None:
    ...


def from_structurefactor() -> None:
    ...


def test_from_gemmi(ccp4_map: gemmi.Ccp4Map) -> None:
    resolution = 1.0
    rsmap = Map.from_gemmi(ccp4_map=ccp4_map, high_resolution_limit=resolution)
    assert len(rsmap) > 0
