import numpy as np
import gemmi as gm
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


def map_from_Fs(dataset, Fs, phis, map_res):
    """
    Return a GEMMI CCP4 map object from an rs.Dataset object

    Parameters :

    dataset  : rs.Dataset of interest
    Fs, phis : (str) and (str) labels for amplitudes and phases to be used
    map_res  : (float) to determine map spacing resolution

    """

    mtz = dataset.to_gemmi()
    ccp4 = gm.Ccp4Map()
    ccp4.grid = mtz.transform_f_phi_to_map(
        "{}".format(Fs), "{}".format(phis), sample_rate=map_res
    )
    ccp4.update_ccp4_header(2, True)

    return ccp4
