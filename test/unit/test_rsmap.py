
import pytest
import numpy as np
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

    
