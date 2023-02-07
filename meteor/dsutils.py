
import numpy as np
import gemmi as gm
import reciprocalspaceship as rs
from scipy.stats import binned_statistic


def res_cutoff(df, h_res, l_res) :
    """
    Apply specified low and high resolution cutoffs to rs.Dataset.
    """
    df = df.loc[(df['dHKL'] >= h_res) & (df['dHKL'] <= l_res)]
    return df


def resolution_shells(data, dhkl, n):

    """ Average data in n resolution shells """

    mean_data   = binned_statistic(dhkl, data, statistic='mean', bins=n, range=(np.min(dhkl), np.max(dhkl)))
    bin_centers = (mean_data.bin_edges[:-1] + mean_data.bin_edges[1:]) / 2 

    return bin_centers, mean_data.statistic


def adjust_phi_interval(phi):

    """ Given a set of phases, return the equivalent in -180 <= phi <= 180 interval"""

    for idx, p in enumerate(phi) :
            if 180 < p <= 361:
                phi[idx] = p - 360

        
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
    new_Fs   = df[Fs].copy(deep=True)
    
    negs = np.where(df[Fs]<0)

    df[phases] = adjust_phi_interval(df[phases])

    for i in negs:
        new_phis.iloc[i] = df[phases].iloc[i]+180
        new_Fs.iloc[i]   = np.abs(new_Fs.iloc[i])
    
    new_phis             = adjust_phi_interval(new_phis)

    df_new               = df.copy(deep=True)
    df_new[Fs_new]       = new_Fs
    df_new[Fs_new]       = df_new[Fs_new].astype("SFAmplitude")
    df_new[phases_new]   = new_phis
    df_new[phases_new]   = df_new[phases_new].astype("Phase")
    
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
    ccp4.grid = mtz.transform_f_phi_to_map('{}'.format(Fs), '{}'.format(phis), sample_rate=map_res)
    ccp4.update_ccp4_header(2, True)
    
    return ccp4


def from_gemmi(gemmi_mtz):
    
    """
    Construct DataSet from gemmi.Mtz object
    
    If the gemmi.Mtz object contains an M/ISYM column and contains duplicated
    Miller indices, an unmerged DataSet will be constructed. The Miller indices
    will be mapped to their observed values, and a partiality flag will be
    extracted and stored as a boolean column with the label, ``PARTIAL``.
    Otherwise, a merged DataSet will be constructed.
    If columns are found with the ``MTZInt`` dtype and are labeled ``PARTIAL``
    or ``CENTRIC``, these will be interpreted as boolean flags used to
    label partial or centric reflections, respectively.
    
    Parameters
    ----------
    gemmi_mtz : gemmi.Mtz
        gemmi Mtz object
    
    Returns
    -------
    rs.DataSet
    """
    
    dataset = rs.DataSet(spacegroup=gemmi_mtz.spacegroup, cell=gemmi_mtz.cell)

    # Build up DataSet
    for c in gemmi_mtz.columns:
        dataset[c.label] = c.array
        # Special case for CENTRIC and PARTIAL flags
        if c.type == "I" and c.label in ["CENTRIC", "PARTIAL"]:
            dataset[c.label] = dataset[c.label].astype(bool)
        else:
            dataset[c.label] = dataset[c.label].astype(c.type)
    dataset.set_index(["H", "K", "L"], inplace=True)

    # Handle unmerged DataSet. Raise ValueError if M/ISYM column is not unique
    m_isym = dataset.get_m_isym_keys()
    if m_isym and dataset.index.duplicated().any():
        if len(m_isym) == 1:
            dataset.merged = False
            dataset.hkl_to_observed(m_isym[0], inplace=True)
        else:
            raise ValueError(
                "Only a single M/ISYM column is supported for unmerged data"
            )
    else:
        dataset.merged = True

    return dataset

