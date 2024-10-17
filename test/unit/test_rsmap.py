from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from meteor.rsmap import Map, MapMutabilityError, MissingUncertaintiesError, _assert_is_map
from meteor.testing import assert_phases_allclose
from meteor.utils import filter_common_indices


def test_assert_is_map(noise_free_map: Map) -> None:
    _assert_is_map(noise_free_map, require_uncertainties=True)  # should work

    del noise_free_map["SIGF"]
    _assert_is_map(noise_free_map, require_uncertainties=False)  # should work
    with pytest.raises(MissingUncertaintiesError):
        _assert_is_map(noise_free_map, require_uncertainties=True)

    not_a_map = rs.DataSet(noise_free_map)
    with pytest.raises(TypeError):
        _assert_is_map(not_a_map, require_uncertainties=False)


def test_initialization_leaves_input_unmodified(noise_free_map: Map) -> None:
    dataset = rs.DataSet(noise_free_map).copy()
    assert not isinstance(dataset, Map)

    dataset["new_column"] = dataset["F"].copy()
    new_map = Map(dataset)
    assert "new_column" in dataset.columns
    assert "new_column" not in new_map.columns


def test_amplitude_and_phase_required(noise_free_map: Map) -> None:
    ds = rs.DataSet(noise_free_map)
    Map(ds)  # should be no problem

    with pytest.raises(KeyError):
        Map(ds, phase_column="does_not_exist")

    del ds["F"]
    with pytest.raises(KeyError):
        Map(ds)


def test_reset_index(noise_free_map: Map) -> None:
    modmap = noise_free_map.reset_index()
    assert len(modmap.columns) == len(noise_free_map.columns) + 3


def test_copy(noise_free_map: Map) -> None:
    copy_map = noise_free_map.copy()
    assert isinstance(copy_map, Map)
    assert copy_map is not noise_free_map
    pd.testing.assert_frame_equal(copy_map, noise_free_map)

    # ensure deep copy
    assert copy_map.values is not noise_free_map.values  # noqa: PD011, want to ensure deep copy
    assert np.all(noise_free_map["F"] == copy_map["F"])
    copy_map["F"] += 1.0
    assert np.all(noise_free_map["F"] != copy_map["F"])


def test_copy_non_standard_names(noise_free_map: Map) -> None:
    ds = rs.DataSet(noise_free_map).rename(columns={"F": "amps", "PHI": "phases"})
    non_std_map = Map(ds, amplitude_column="amps", phase_column="phases")
    copy_map = non_std_map.copy()

    assert isinstance(copy_map, Map)
    assert copy_map is not non_std_map
    assert copy_map.values is not non_std_map.values  # noqa: PD011, want to ensure deep copy
    assert "amps" in copy_map.columns
    assert "phases" in copy_map.columns
    pd.testing.assert_frame_equal(copy_map, non_std_map)


def test_filter_common_indices_with_maps(noise_free_map: Map) -> None:
    m1 = noise_free_map
    m2 = noise_free_map.copy()
    m2.drop([m1.index[0]], axis=0, inplace=True)  # remove an index
    assert len(m1) != len(m2)
    f1, f2 = filter_common_indices(m1, m2)
    pd.testing.assert_index_equal(f1.index, f2.index)
    assert len(f1.columns) == 3
    assert len(f2.columns) == 3


def test_verify_type(noise_free_map: Map) -> None:
    dataseries = rs.DataSeries([1, 2, 3])
    assert dataseries.dtype == int
    noise_free_map._verify_type("foo", [int], dataseries, fix=False, cast_fix_to=int)

    with pytest.raises(AssertionError):
        noise_free_map._verify_type("foo", [float], dataseries, fix=False, cast_fix_to=float)

    output = noise_free_map._verify_type("foo", [float], dataseries, fix=True, cast_fix_to=float)
    assert output.dtype == float


@pytest.mark.parametrize("fix", [False, True])
def test_verify_types(noise_free_map: Map, fix: bool) -> None:
    functions_to_test = [
        noise_free_map._verify_amplitude_type,
        noise_free_map._verify_phase_type,
        noise_free_map._verify_uncertainty_type,
    ]

    expected_fixed_types = [
        rs.StructureFactorAmplitudeDtype(),
        rs.PhaseDtype(),
        rs.StandardDeviationDtype(),
    ]

    dataseries_to_use = [
        noise_free_map.amplitudes,
        noise_free_map.phases,
        noise_free_map.uncertainties,
        rs.DataSeries(np.arange(10)),
    ]

    # success_matrix[i][j] = functions_to_test[i] expected to pass on dataseries_to_use[j]
    success_matrix = [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
    ]

    for i, fxn in enumerate(functions_to_test):
        for j, ds in enumerate(dataseries_to_use):
            if success_matrix[i][j] or fix:
                output = fxn(ds, fix=fix)  # should pass
                if fix:
                    assert output.dtype == expected_fixed_types[i]
            else:
                with pytest.raises(AssertionError):
                    fxn(ds, fix=fix)


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


def test_get_hkls_consistent_with_reciprocalspaceship(noise_free_map: Map) -> None:
    meteor_hkl = noise_free_map.get_hkls()
    rs_hkl = rs.DataSet(noise_free_map).get_hkls()
    np.testing.assert_array_equal(meteor_hkl, rs_hkl)


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
    assert isinstance(noise_free_map.amplitudes, rs.DataSeries)
    assert isinstance(noise_free_map.phases, rs.DataSeries)
    assert isinstance(noise_free_map.uncertainties, rs.DataSeries)

    noise_free_map.amplitudes *= 2.0
    noise_free_map.phases *= 2.0
    noise_free_map.uncertainties *= 2.0


def test_has_uncertainties(noise_free_map: Map) -> None:
    assert noise_free_map.has_uncertainties
    del noise_free_map["SIGF"]
    assert not noise_free_map.has_uncertainties


@pytest.mark.filterwarnings("ignore:Pandas doesn't allow columns to be created via a new attribute")
def test_set_uncertainties() -> None:
    test_map = Map.from_dict(
        {"F": rs.DataSeries([2.0, 3.0, 4.0]), "PHI": rs.DataSeries([0.0, 0.0, 0.0])},
    )

    assert not test_map.has_uncertainties
    with pytest.raises(AttributeError):
        _ = test_map.uncertainties

    # this would normally generate the suppressed warning, but we are more strict and raise
    with pytest.raises(AttributeError):
        test_map.uncertainties = rs.DataSeries([2.0, 3.0, 4.0])

    test_map.set_uncertainties(rs.DataSeries([1.0, 1.0, 1.0]))
    assert test_map.has_uncertainties
    assert len(test_map.uncertainties) == 3


def test_misconfigured_columns() -> None:
    test_map = Map.from_dict(
        {"F": rs.DataSeries([2.0, 3.0, 4.0]), "PHI": rs.DataSeries([0.0, 0.0, 0.0])},
    )
    del test_map["F"]
    with pytest.raises(RuntimeError):
        test_map.set_uncertainties(rs.DataSeries([2.0, 3.0, 4.0]))


def test_complex_amplitudes_smoke(noise_free_map: Map) -> None:
    c_array = noise_free_map.complex_amplitudes
    assert isinstance(c_array, np.ndarray)
    assert np.issubdtype(c_array.dtype, np.complexfloating)


def test_complex_amplitudes() -> None:
    index = pd.Index(np.arange(4))
    amp = rs.DataSeries(np.ones(4), index=index, name="F")
    phase = rs.DataSeries(np.arange(4) * 90.0, index=index, name="PHI")

    ds = rs.concat([amp, phase], axis=1)
    rsmap = Map(ds)

    expected = np.array([1.0, 0.0, -1.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0, -1.0])
    np.testing.assert_almost_equal(rsmap.complex_amplitudes, expected)


def test_from_dataset(noise_free_map: Map) -> None:
    map_as_dataset = rs.DataSet(noise_free_map)
    map2 = Map(
        map_as_dataset,
        amplitude_column=noise_free_map._amplitude_column,
        phase_column=noise_free_map._phase_column,
        uncertainty_column=noise_free_map._uncertainty_column,
    )
    pd.testing.assert_frame_equal(noise_free_map, map2)


def test_to_structurefactor(noise_free_map: Map) -> None:
    c_dataseries = noise_free_map.to_structurefactor()
    c_array = noise_free_map.complex_amplitudes
    np.testing.assert_almost_equal(c_dataseries.to_numpy(), c_array)


def from_structurefactor(noise_free_map: Map) -> None:
    map2 = Map.from_structurefactor(noise_free_map.complex_amplitudes, index=noise_free_map.index)
    pd.testing.assert_frame_equal(noise_free_map, map2)


def test_to_ccp4_map(noise_free_map: Map) -> None:
    ccp4_map = noise_free_map.to_ccp4_map(map_sampling=3)
    assert ccp4_map.grid.shape == (30, 30, 30)


def test_gemmi_mtz_roundtrip(noise_free_map: Map) -> None:
    gemmi_mtz = noise_free_map.to_gemmi()
    assert isinstance(gemmi_mtz, gemmi.Mtz)
    map2 = Map.from_gemmi(gemmi_mtz)
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


def test_to_and_from_mtz_file(noise_free_map: Map, tmp_path: Path) -> None:
    file_path = tmp_path / "tmp.mtz"
    noise_free_map.write_mtz(file_path)
    loaded = Map.read_mtz_file(file_path)
    pd.testing.assert_frame_equal(noise_free_map, loaded)
