import pytest
import numpy as  np
import reciprocalspaceship as rs
from   pathlib import Path
from   meteor  import dsutils

@pytest.fixture
def dummy_mtz():
    data = np.array([1, 2, 3, 4, 5])
    dhkl = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    dataset = rs.DataSet({"data": data, "dHKL": dhkl})
    return dataset

def test_res_cutoff(dummy_mtz):
    df = dummy_mtz
    h_res = 1.2
    l_res = 1.4

    filtered_df = dsutils.res_cutoff(df, h_res, l_res)

    assert len(filtered_df) == 2
    assert filtered_df["data"].tolist() == [2, 3]

def test_res_cutoff_raises_error(dummy_mtz):
    df = dummy_mtz
    h_res = 1.5
    l_res = 1.2

    with pytest.raises(ValueError):
        dsutils.res_cutoff(df, h_res, l_res)

def test_resolution_shells(dummy_mtz):
    data = dummy_mtz["data"]
    dhkl = dummy_mtz["dHKL"]
    n = 2

    bin_centers, mean_data = dsutils.resolution_shells(data, dhkl, n)

    assert len(bin_centers) == n
    assert len(mean_data.statistic) == n

def test_resolution_shells_raises_error(dummy_mtz):
    data = dummy_mtz["data"]
    dhkl = dummy_mtz["dHKL"]
    n = 10

    with pytest.raises(ValueError):
        dsutils.resolution_shells(data, dhkl, n)