from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.rsmap import Map, MapMutabilityError
from meteor.testing import assert_phases_allclose
from meteor.utils import filter_common_indices


def test_assert_is_map() -> None:
    raise NotImplementedError  # TODO: implement


def test_initialization_leaves_input_unmodified(noise_free_map: Map) -> None:
    dataset = rs.DataSet(noise_free_map).copy()
    assert not isinstance(dataset, Map)

    dataset["new_column"] = dataset["F"].copy()
    new_map = Map(dataset)
    assert "new_column" in dataset.columns
    assert "new_column" not in new_map.columns


def test_copy(noise_free_map: Map) -> None:
    copy_map = noise_free_map.copy()
    assert isinstance(copy_map, Map)
    pd.testing.assert_frame_equal(copy_map, noise_free_map)


def test_filter_common_indices_with_maps(noise_free_map: Map) -> None:
    m1 = noise_free_map
    m2 = noise_free_map.copy()
    m2.drop([m1.index[0]], axis=0, inplace=True)  # remove an index
    assert len(m1) != len(m2)
    f1, f2 = filter_common_indices(m1, m2)
    pd.testing.assert_index_equal(f1.index, f2.index)
    assert len(f1.columns) == 3
    assert len(f2.columns) == 3


def test_verify_types() -> None:
    # _verify_type
    # _verify_amplitude_type
    # _verify_phase_type
    # _verify_uncertainty_type
    raise NotImplementedError  # TODO: implement


def test_setitem(noise_free_map: Map, noisy_map: Map) -> None:
    noisy_map.amplitudes = noise_free_map.amplitudes
    noisy_map.phases = noise_free_map.phases
    noisy_map.set_uncertainties(noise_free_map.uncertainties)  # should work even if already set


def test_unallowed_setitem_disabled(noise_free_map: Map) -> None:
    with pytest.raises(MapMutabilityError):
        noise_free_map["unallowed_column_name"] = noise_free_map.amplitudes


def test_insert_disabled(noise_free_map: Map) -> None:
    position = 0
    column = "foo"
    value = [0, 1]
    with pytest.raises(MapMutabilityError):
        noise_free_map.insert(position, column, value)


def test_drop_columns_disabled(noise_free_map: Map) -> None:
    noise_free_map.drop([0, 0, 0], axis=0)
    with pytest.raises(MapMutabilityError):
        noise_free_map.drop("F", axis=1)
    with pytest.raises(MapMutabilityError):
        noise_free_map.drop(columns=["F"])


def test_get_hkls(noise_free_map: Map) -> None:
    hkl = noise_free_map.get_hkls()
    assert len(hkl.shape) == 2
    assert hkl.shape[0] > 0
    assert hkl.shape[1] == 3


def test_compute_dhkl(noise_free_map: Map) -> None:
    d_hkl = noise_free_map.compute_dHKL()
    assert np.max(d_hkl) == 10.0
    assert np.min(d_hkl) == 1.0
    assert d_hkl.shape == noise_free_map.amplitudes.shape


def test_resolution_limits(random_difference_map: Map) -> None:
    dmax, dmin = random_difference_map.resolution_limits
    assert dmax == 10.0
    assert dmin == 1.0


def test_get_set_fixed_columns(noise_free_map: Map) -> None:
    assert isinstance(noise_free_map.amplitudes, rs.DataSet)
    assert isinstance(noise_free_map.phases, rs.DataSet)
    assert isinstance(noise_free_map.uncertainties, rs.DataSet)

    noise_free_map.amplitudes *= 2.0
    noise_free_map.phases *= 2.0
    noise_free_map.uncertainties *= 2.0


def test_has_uncertainties(noise_free_map: Map) -> None:
    assert noise_free_map.has_uncertainties
    del noise_free_map["SIGF"]
    assert not noise_free_map.has_uncertainties


def test_set_uncertainties() -> None:
    test_map = Map.from_dict(
        {"F": rs.DataSeries([2.0, 3.0, 4.0]), "PHI": rs.DataSeries([0.0, 0.0, 0.0])},
    )

    assert not test_map.has_uncertainties
    with pytest.raises(AttributeError):
        _ = test_map.uncertainties

    test_map.set_uncertainties(rs.DataSeries([1.0, 1.0, 1.0]))
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
    map2 = Map.from_structurefactor(noise_free_map.complex_amplitudes, index=noise_free_map.index)
    pd.testing.assert_frame_equal(noise_free_map, map2)


def test_from_ccp4_map(ccp4_map: gemmi.Ccp4Map) -> None:
    resolution = 1.0
    rsmap = Map.from_ccp4_map(ccp4_map, high_resolution_limit=resolution)
    assert len(rsmap) > 0


@pytest.mark.parametrize("map_sampling", [1, 2, 2.25, 3, 5])
def test_ccp4_map_round_trip(
    map_sampling: int,
    random_difference_map: Map,
) -> None:
    realspace_map = random_difference_map.to_ccp4_map(map_sampling=map_sampling)

    _, dmin = random_difference_map.resolution_limits
    output_coefficients = Map.from_ccp4_map(realspace_map, high_resolution_limit=dmin)

    random_difference_map.canonicalize_amplitudes()
    output_coefficients.canonicalize_amplitudes()

    pd.testing.assert_series_equal(
        random_difference_map.amplitudes,
        output_coefficients.amplitudes,
        atol=1e-3,
    )
    assert_phases_allclose(
        random_difference_map.phases.to_numpy(),
        output_coefficients.phases.to_numpy(),
    )


def test_from_mtz_file(noise_free_map: Map, tmp_path: Path) -> None:
    file_path = tmp_path / "tmp.mtz"
    noise_free_map.write_mtz(file_path)
    loaded = Map.from_mtz_file(file_path)
    pd.testing.assert_frame_equal(noise_free_map, loaded)
