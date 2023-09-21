
import numpy as np
import gemmi as gm

from . import scale
from . import io


def make_map(data, grid_size, cell, space_group) :

    """
    Create a GEMMI map object from data and grid information.
    
    Args :
    
    data (ndarray)    : containing map values
    grid_size (list)  : specifying grid dimensions for the map
    cell, space_group : (list of 6 floats) and (str) 
    
    Returns :

    GEMMI CCP4 map object

    """

    og = gm.Ccp4Map()
    
    og.grid = gm.FloatGrid(data)
    og.grid.set_unit_cell(gm.UnitCell(*cell))
    og.grid.set_size(*grid_size)
    og.grid.spacegroup = gm.find_spacegroup_by_name(space_group)
    og.grid.symmetrize_max()
    og.update_ccp4_header()
    
    return og


def compute_weights(df, sigdf, alpha=0.0):
    
    """
    Compute weights for each structure factor based on DeltaF and its uncertainty.
    Parameters
    ----------
    df : series-like or array-like
        Array of DeltaFs (difference structure factor amplitudes)
    sigdf : series-like or array-like
        Array of SigDeltaFs (uncertainties in difference structure factor amplitudes)
    """
    w = (1 + (sigdf**2 / (sigdf**2).mean()) + alpha*(df**2 / (df**2).mean()))
    return w**-1


def find_w_diffs(mtz, a=0.00, Nbg=1.00):

    """ 
    
    Calculate weighted difference structure factors from a pre-scaled MTZ.

    Args:
        mtz (rsDataset)      : Output from the `scale_toFCalcs` function.
        a (float, optional)  : Alpha weighting parameter q-weighting. Default is 0.00.
        Nbg (float, optional): If making a background subtracted map. Default is 1.00.

    Returns:
        rsDataset: MTZ with added column for weighted differences.
        np.ndarray: Weights applied to each structure factor difference (1D array).

    Raises:
        ValueError: If required columns 'SIGF_on_s', 'SIGF_off_s', 'scaled_on', 
        or 'scaled_off' are missing in `mtz`.
    """
    required_columns = ['SIGF_on_s', 'SIGF_off_s', 'scaled_on', 'scaled_off']
    missing_columns = [col for col in required_columns if col not in mtz.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in 'mtz': {', '.join(missing_columns)}")

    sig_diffs  = np.sqrt(mtz["SIGF_on_s"]**2 + (mtz["SIGF_off_s"])**2)
    ws = compute_weights(mtz["scaled_on"] - Nbg * mtz["scaled_off"], sig_diffs, alpha=a)
    
    mtz["DF"]  = (mtz["scaled_on"] - Nbg * mtz["scaled_off"])
    mtz["DF"]  = mtz["DF"].astype("SFAmplitude")
    mtz["WDF"] = ws * (mtz["scaled_on"] - Nbg * mtz["scaled_off"])
    mtz["WDF"] = mtz["WDF"].astype("SFAmplitude")
    mtz.infer_mtz_dtypes(inplace=True)

    return mtz, ws


def scale_toFCalcs(mtz, labels, pdb, high_res):

    """
    Scale two sets of data (on and off) to calculated structure factors (FCalcs).

        Args:
            mtz (rsDataset)     : Contains both sets of data.
            labels (MtzLabels)  : Labels for the structure factors and errors.
            pdb (str)           : Path for the PDB file to calculate FCalcs.
            high_res (float)    : High resolution limit for FCalcs.

        Returns:
            rsDataset: MTZ with added columns.
    """

    calcs     = io.get_Fcalcs(pdb, high_res-0.05)['FC']
    calcs     = calcs[calcs.index.isin(mtz.index)]

    scaled_on   = scale.scale_aniso(np.array(calcs), np.array(mtz[labels.Fon]), 
                                    np.array(list(mtz.index)))
    scaled_off  = scale.scale_aniso(np.array(calcs), np.array(mtz[labels.Foff]), 
                                    np.array(list(mtz.index)))
    
    mtz["scaled_on"]   = scaled_on[-1]
    mtz["scaled_off"]  = scaled_off[-1]
    mtz["SIGF_on_s"]   = (scaled_on[0].x[0]  
                          * np.exp(scaled_on[1]))  * mtz[labels.SigFon]
    mtz["SIGF_off_s"]  = (scaled_off[0].x[0] 
                          * np.exp(scaled_off[1])) * mtz[labels.SigFoff] 
    
    mtz.infer_mtz_dtypes(inplace=True)
    
    return mtz




  