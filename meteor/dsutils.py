
import numpy as np
import gemmi as gm
import reciprocalspaceship as rs
from scipy.stats import binned_statistic


def res_cutoff(df, h_res, l_res) :
    """
    
    Apply specified low and high resolution cutoffs to the rs.Dataset.

    Args:
        df (rs.DataFrame): The input rs.Dataset to filter.
        h_res (float): The high resolution cutoff value.
        l_res (float): The low resolution cutoff value.

    Returns:
        rs.DataFrame: The filtered rs.Dataset based on the resolution cutoffs.

    """
    df = df.loc[(df['dHKL'] >= h_res) & (df['dHKL'] <= l_res)]
    return df


def resolution_shells(data, dhkl, n):

    """
    Average data in n resolution shells.

    Args:
        data (array-like): The mtz data to be averaged.
        dhkl (array-like): The values of dHKL (resolution) associated 
                            with each data point.
        n (int): The number of resolution shells to create.

    Returns:
        tuple: A tuple containing two arrays.
            - bin_centers (ndarray): The centers of the resolution shells.
            - mean_data (BinnedStatisticResult): The binned statistic result 
              containing the averaged data.

    Raises:
        ValueError: If the lengths of data and dhkl do not match.
                    If n is greater than the number of unique values in dhkl.
                    If no data points fall within the resolution shells.

    Note:
        The binned_statistic function used in this function is the scipy.stats module.
    """

    if len(data) != len(dhkl):
        raise ValueError("Lengths of 'data' and 'dhkl' must be equal.")

    if n > len(np.unique(dhkl)):
        raise ValueError("'n' is greater than the number of unique values in 'dhkl'.")

    mean_data = binned_statistic(dhkl, data, statistic='mean', bins=n, 
                                 range=(np.min(dhkl), np.max(dhkl)))
    bin_centers = (mean_data.bin_edges[:-1] + mean_data.bin_edges[1:]) / 2

    if np.isnan(mean_data.statistic).all():
        raise ValueError("No data points fall within the resolution shells.")

    return bin_centers, mean_data


def adjust_phi_interval(phi):
    """
    Adjust the phase values to the equivalent in the -180 <= phi <= 180 interval.

    Args:
        phi (array-like): The set of phase values.

    Returns:
        ndarray: The adjusted phase values in the -180 <= phi <= 180 interval.

    Raises:
        AssertionError: If the minimum value of phi is less than -181.
        AssertionError: If the maximum value of phi is greater than 181.

    """
    phi = phi % 360
    phi[phi > 180] -= 360

    assert np.min(phi) >= -181, "Minimum value of phi is less than -181."
    assert np.max(phi) <= 181, "Maximum value of phi is greater than 181."

    return phi




def positive_Fs(df, phases, Fs):
    """
    Convert between an MTZ format where difference structure factor amplitudes 
    are saved as both positive and negative,
    to a format where they are only positive.

    Args:
        df (rs.Dataset) : The input rs.Dataset from the MTZ of interest.
        phases (str)    : Label for the phases column in the original MTZ.
        Fs (str)        : Label for the amplitudes column in the original MTZ.

    Returns:
        rs.Dataset: The updated rs.Dataset with new labels.

    Raises:
        ValueError: If any of the input columns (phases, Fs) are not present in df.
        TypeError:  If the input columns (phases, Fs) have incorrect data types.

    """
    if any(col not in df.columns for col in [phases, Fs]):
        raise ValueError("One or more input columns not present in df.")

    if not all(df[col].dtype == dtype for 
               col, dtype in [(Fs, "SFAmplitude"), (phases, "Phase")]):
        raise TypeError("Incorrect data type for one or more input columns."
                        "Expected 'SFAmplitude' for amplitudes and 'Phase' for phases.")

    df_new = df.copy()

    # Adjust phases and Fs for negative Fs values
    negs = df[Fs] < 0
    df_new.loc[negs, phases] = adjust_phi_interval(df[phases]) + 180
    df_new.loc[negs, Fs] = np.abs(df[Fs])

    # Adjust the new phases column and change data types
    df_new[phases+"_pos"] = adjust_phi_interval(df_new[phases])
    df_new[Fs+"_pos"] = df_new[Fs].astype("SFAmplitude")
    df_new[phases+"_pos"] = df_new[phases+"_pos"].astype("Phase")

    return df_new



def map_from_Fs(dataset, Fs, phis, map_res):
    
    """
    Return a GEMMI CCP4 map object from an rs.Dataset object.

    Args:
        dataset (rs.Dataset): The rs.Dataset of interest.
        Fs (str): Label for the amplitudes column to be used.
        phis (str): Label for the phases column to be used.
        map_res (float): Map spacing resolution to determine the map resolution.

    Returns:
        gm.Ccp4Map: The GEMMI CCP4 map object.  
    """
    
    mtz = dataset.to_gemmi()
    ccp4 = gm.Ccp4Map()
    ccp4.grid = mtz.transform_f_phi_to_map('{}'.format(Fs), '{}'.format(phis), 
                                           sample_rate=map_res)
    ccp4.update_ccp4_header(2, True)
    
    return ccp4

def from_dataframe(df, pdb_params, indices, types):

    ds = rs.DataSet(spacegroup=pdb_params.Xtal.SpaceGroup, 
                    cell=pdb_params.Xtal.Cell) 

    # Build up DataSet
    for idx, (label, data) in enumerate(df.iteritems()):
        ds[label] = data
        ds[label] = ds[label].astype(types[idx])
    
    ds.index = indices 
    return ds


def from_gemmi(gemmi_mtz):
    
    """
    Note: original function from from reciprocalspaceship 
    
    Construct DataSet from gemmi.Mtz object

    If the gemmi.Mtz object contains an M/ISYM column and contains duplicated
    Miller indices, an unmerged DataSet will be constructed. The Miller indices
    will be mapped to their observed values, and a partiality flag will be
    extracted and stored as a boolean column with the label, "PARTIAL".
    Otherwise, a merged DataSet will be constructed.
    If columns are found with the "MTZInt" dtype and are labeled "PARTIAL"
    or "CENTRIC", these will be interpreted as boolean flags used to
    label partial or centric reflections, respectively.

    Parameters:
        gemmi_mtz (gemmi.Mtz): gemmi Mtz object

    Returns:
        rs.DataSet: The constructed DataSet.

    Raises:
        ValueError: If the gemmi.Mtz object contains multiple M/ISYM columns,
                    or if the M/ISYM column is not unique for unmerged data.
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

