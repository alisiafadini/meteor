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


def screen_alpha_weight(
    data,
    F_on,
    F_off,
    SIGF_on,
    SIGF_off,
    philabel,
    refpdb,
    path,
    name,
    map_res,
    center,
    radius=8.0,
    hres=None,
    percent=0.03,
    flags=None,
):
    """
    This is a work in progress

    """
    # TO ADD: assertion or test for no negative amplitudes in F_on or F_off

    # if Rfree flags were specified, use these
    if flags is not None:
        _, _, choose_test = validate.make_test_set(
            data, percent, F_off, name, path, flags
        )  # choose_test is Boolean array to select free (test) reflections from entire set

    # Keep 3% of reflections for test set
    else:
        _, _, choose_test = validate.make_test_set(data, percent, F_off, name, path)

    alphas = np.linspace(0, 1.0, 20)
    noweight, _ = find_w_diffs(
        data, F_on, F_off, SIGF_on, SIGF_off, refpdb, hres, path, 0.0
    )
    noweight_pos = dsutils.positive_Fs(
        noweight, philabel, "DF", "ogPhis_pos", "ogFs_pos"
    )

    errors = []
    entropies = []
    datasets = []

    for a in tqdm(alphas):
        diffs, _ = find_w_diffs(
            data.copy(deep=True), F_on, F_off, SIGF_on, SIGF_off, refpdb, hres, path, a
        )

        # make DED map, find negentropy
        fit_map = dsutils.map_from_Fs(diffs, "WDF", philabel, map_res)

        # mask chromophore region for negentropy
        reg_mask = mask.get_mapmask(fit_map.grid, center, radius)
        loc_reg = reg_mask.flatten().astype(bool)

        fit_map_arr = np.array(fit_map.grid).flatten()[loc_reg]
        # fit_map_arr = np.array(fit_map.grid).flatten()
        error = kurtosis(fit_map_arr)
        # entropy = differential_entropy(fit_map_arr)
        entropy = validate.negentropy(fit_map_arr)

        # backcalculate all Fs
        if hres is not None:
            diffs_all = dsutils.from_gemmi(io.map2mtz(fit_map, hres))
        else:
            diffs_all = dsutils.from_gemmi(
                io.map2mtz(fit_map, np.min(noweight.compute_dHKL()["dHKL"]))
            )

        diffs_all = diffs_all[diffs_all.index.isin(noweight_pos.index)]
        # error         = np.sum(np.array(noweight_pos["ogFs_pos"][np.invert(choose_test)]) - np.array(diffs_all["FWT"][np.invert(choose_test)])) ** 2
        # print(np.array(noweight_pos["ogFs_pos"][np.invert(choose_test)])[0:20], np.array(diffs_all["FWT"][np.invert(choose_test)])[0:20])
        # print(np.array(noweight_pos["ogFs_pos"][choose_test])[0:20], np.array(diffs_all["FWT"][choose_test])[0:20])

        # error         = np.sum(np.array(noweight_pos["ogFs_pos"][np.invert(choose_test)]) - np.array(diffs_all["FWT"][np.invert(choose_test)])) ** 2
        # print(diffs_all[choose_test])
        entropies.append(entropy)
        datasets.append(diffs_all)
        errors.append(error)

    print("Best Weighting Alpha Found is ", alphas[np.argmin(errors)])

    return datasets[np.argmin(errors)], errors, entropies
