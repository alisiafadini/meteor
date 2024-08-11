import numpy as np
import gemmi as gm
from tqdm import tqdm
from meteor import dsutils, validate, mask
from scipy.stats import kurtosis

from . import scale
from . import io


def make_map(data, grid_size, cell, space_group):
    """
    Create a GEMMI map object from data and grid information.

    Parameters :

    data              : (numpy array)
    grid_size         : (list) specifying grid dimensions for the map
    cell, space_group : (list) and (str)

    Returns :

    GEMMI CCP4 map object

    """
    og = gm.Ccp4Map()

    og.grid = gm.FloatGrid(data)
    og.grid.set_unit_cell(
        gm.UnitCell(cell[0], cell[1], cell[2], cell[3], cell[4], cell[5])
    )
    og.grid.set_size(grid_size[0], grid_size[1], grid_size[2])
    og.grid.spacegroup = gm.find_spacegroup_by_name(space_group)
    og.grid.symmetrize_max()
    og.update_ccp4_header()

    return og


def compute_weights(df, sigdf, alpha):
    """
    Compute weights for each structure factor based on DeltaF and its uncertainty.
    Parameters
    ----------
    df : series-like or array-like
        Array of DeltaFs (difference structure factor amplitudes)
    sigdf : series-like or array-like
        Array of SigDeltaFs (uncertainties in difference structure factor amplitudes)
    """
    w = 1 + (sigdf**2 / (sigdf**2).mean()) + alpha * (df**2 / (df**2).mean())
    return w**-1


def find_w_diffs(mtz, Fon, Foff, SIGon, SIGoff, pdb, high_res, path, a, Nbg=1.00):
    """

    Calculate weighted difference structure factors from a reference structure and input mtz.

    Parameters :

    1. MTZ, Fon, Foff, SIGFon, SIGFoff           : (rsDataset) with specified structure factor and error labels (str)
    2. pdb                                       : reference pdb file name (str)
    3. highres                                   : high resolution cutoff for map generation (float)
    4. path                                      : path of directory where to store any files (string)
    5. a                                         : alpha weighting parameter q-weighting (float)
    6. Nbg                                       : background subtraction value if making a background subtracted map (float – default=1.00)


    Returns :

    1. mtz                                      : (rs-Dataset) of original mtz + added column for weighted differences
    2. ws                                       : weights applied to each structure factor difference (1D array)

    """
    calcs = io.get_Fcalcs(pdb, high_res, path)["FC"]
    calcs = calcs[calcs.index.isin(mtz.index)]
    # calcs = mtz["FC"]
    mtx_on, t_on, scaled_on = scale.scale_aniso(
        np.array(calcs), np.array(mtz[Fon]), np.array(list(mtz.index))
    )
    mtx_off, t_off, scaled_off = scale.scale_aniso(
        np.array(calcs), np.array(mtz[Foff]), np.array(list(mtz.index))
    )

    mtz["scaled_on"] = scaled_on
    mtz["scaled_off"] = scaled_off
    mtz["SIGF_on_s"] = (mtx_on.x[0] * np.exp(t_on)) * mtz[SIGon]
    mtz["SIGF_off_s"] = (mtx_off.x[0] * np.exp(t_off)) * mtz[SIGoff]
    # mtz = mtz.compute_dHKL()
    # qs = 1/(2*mtz['dHKL'])
    # c_on, b_on, on_s     = scale.scale_iso(np.array(calcs), np.array(mtz[Fon]),  np.array(mtz['dHKL']))
    # c_off, b_off, off_s = scale.scale_iso(np.array(calcs), np.array(mtz[Foff]), np.array(mtz['dHKL']))

    # mtz["scaled_on"] = on_s
    # mtz["scaled_off"] = off_s
    # mtz["SIGF_on_s"]      = (c_on  * np.exp(-b_on*(qs**2)))  * mtz[SIGon]
    # mtz["SIGF_off_s"]     = (c_off * np.exp(-b_off*(qs**2))) * mtz[SIGoff]

    sig_diffs = np.sqrt(mtz["SIGF_on_s"] ** 2 + (mtz["SIGF_off_s"]) ** 2)
    ws = compute_weights(mtz["scaled_on"] - Nbg * mtz["scaled_off"], sig_diffs, alpha=a)
    mtz["DF"] = mtz["scaled_on"] - Nbg * mtz["scaled_off"]
    mtz["DF"] = mtz["DF"].astype("SFAmplitude")
    mtz["WDF"] = ws * (mtz["scaled_on"] - Nbg * mtz["scaled_off"])
    mtz["WDF"] = mtz["WDF"].astype("SFAmplitude")
    mtz.infer_mtz_dtypes(inplace=True)

    return mtz, ws



