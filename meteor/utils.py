import numpy as np
import gemmi as gm
import reciprocalspaceship as rs
from scipy.stats import binned_statistic


def res_cutoff(df, h_res, l_res):
    """
    Apply specified low and high resolution cutoffs to rs.Dataset.
    """
    df = df.loc[(df["dHKL"] >= h_res) & (df["dHKL"] <= l_res)]
    return df


def resolution_shells(data, dhkl, n):
    """Average data in n resolution shells"""

    mean_data = binned_statistic(
        dhkl, data, statistic="mean", bins=n, range=(np.min(dhkl), np.max(dhkl))
    )
    bin_centers = (mean_data.bin_edges[:-1] + mean_data.bin_edges[1:]) / 2

    return bin_centers, mean_data.statistic


def adjust_phi_interval(phi):
    """Given a set of phases, return the equivalent in -180 <= phi <= 180 interval"""

    phi = phi % 360
    phi[phi > 180] -= 360

    assert np.min(phi) >= -181
    assert np.max(phi) <= 181

    return phi


def positive_Fs(df, phases, Fs, phases_new, Fs_new):
    """
    Convert between an MTZ format where difference structure factor amplitudes are saved as both positive and negative, to format where they are only positive.

    Parameters :

    df                 : (rs.Dataset) from MTZ of interest
    phases, Fs         : (str) labels for phases and amplitudes in original MTZ
    phases_new, Fs_new : (str) labels for phases and amplitudes in new MTZ


    Returns :

    rs.Dataset with new labels

    """

    new_phis = df[phases].copy(deep=True)
    new_Fs = df[Fs].copy(deep=True)

    negs = np.where(df[Fs] < 0)

    df[phases] = adjust_phi_interval(df[phases])

    for i in negs:
        new_phis.iloc[i] = df[phases].iloc[i] + 180
        new_Fs.iloc[i] = np.abs(new_Fs.iloc[i])

    new_phis = adjust_phi_interval(new_phis)

    df_new = df.copy(deep=True)
    df_new[Fs_new] = new_Fs
    df_new[Fs_new] = df_new[Fs_new].astype("SFAmplitude")
    df_new[phases_new] = new_phis
    df_new[phases_new] = df_new[phases_new].astype("Phase")

    return df_new


def compute_map_from_coefficients(
    *,
    map_coefficients: rs.DataSet,
    amplitude_label: str,
    phase_label: str,
    map_sampling: int,
) -> gm.Ccp4Map:
    map_coefficients_gemmi_format = map_coefficients.to_gemmi()
    ccp4_map = gm.Ccp4Map()
    ccp4_map.grid = map_coefficients_gemmi_format.transform_f_phi_to_map(
        amplitude_label, phase_label, sample_rate=map_sampling
    )
    ccp4_map.update_ccp4_header()

    return ccp4_map


def compute_coefficients_from_map(
    *,
    ccp4_map: gm.Ccp4Map,
    high_resolution_limit: float,
    amplitude_label: str,
    phase_label: str,
) -> rs.DataSet:
    # to ensure we include the final shell of reflections, add a small buffer to the resolution
    high_resolution_buffer = 0.05

    gemmi_structure_factors = gm.transform_map_to_f_phi(ccp4_map.grid, half_l=False)
    data = gemmi_structure_factors.prepare_asu_data(
        dmin=high_resolution_limit - high_resolution_buffer, with_sys_abs=True
    )

    mtz = gm.Mtz(with_base=True)
    mtz.spacegroup = gemmi_structure_factors.spacegroup
    mtz.set_cell_for_all(gemmi_structure_factors.unit_cell)
    mtz.add_dataset("FromMap")
    mtz.add_column(amplitude_label, "F")
    mtz.add_column(phase_label, "P")
    mtz.set_data(data)
    mtz.switch_to_asu_hkl()

    return rs.DataSet.from_gemmi(mtz)
