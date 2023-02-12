import numpy as np
from tqdm import tqdm
from skimage.restoration import denoise_tv_chambolle

from . import maps
from . import validate
from . import dsutils
from . import io

def TV_filter(map, l, grid_size, cell, space_group):
    
    """
    Apply TV filtering to a Gemmi map object. Compute negentropy for denoised array.

    Parameters :

    map           : (GEMMI map object)
    l             : (float) lambda – regularization parameter to be used in filtering.
    grid_size         : (list) specifying grid dimensions for the map
    cell, space_group : (list) and (str)

    Returns :

    Denoised map (GEMMI object) and associated negentropy (float)
    
    """

    TV_arr     = denoise_tv_chambolle(np.array(map.grid), eps=0.00000005, weight=l, max_num_iter=50)
    entropy    = validate.negentropy(TV_arr.flatten())
    TV_map     = maps.make_map(TV_arr, grid_size, cell, space_group)

    return TV_map, entropy


def TV_iteration(mtz, diffs, phi_diffs, Fon, Foff, phicalc, map_res, cell, space_group, flags, l, highres, ws):

    """
    Go through one iteration of TV denoising Fon-Foff map + projection onto measured Fon magnitude 
    to obtain new phases estimates for the difference structure factors vector.

    Parameters :

    1. MTZ, diffs, phi_diffs, Fon, Foff, phicalc : (rsDataset) with specified structure factor and phase labels (str)
    2. map_res                                   : (float) spacing for map generation
    3. cell, space group                         : (array) and (str) 
    4. flags                                     : (boolean array) where True marks reflections to keep for test set
    5. l                                         : (float) weighting parameter for TV denoising
    6. highres                                   : (float) high resolution cutoff for map generation
    
    Returns :
    
    1. new amps, new phases                      : (1D-array) for new difference amplitude and phase estimates after the iteration 
    2. test_proj_errror                          : (float) the magnitude of the projection of the TV difference vector onto Fon for the test set
    3. entropy                                   : (float) negentropy of TV denoised map
    4. phase_change                              : (float) mean change in phases between input (phi_diffs) and output (new_phases) arrays

    """

    # Fon - Foff and TV denoise
    mtz           = dsutils.positive_Fs(mtz, phi_diffs, diffs, "phases-pos", "diffs-pos")
    fit_mtz       = mtz[np.invert(flags)] 
    
    fit_map             = dsutils.map_from_Fs(fit_mtz, "diffs-pos", "phases-pos", map_res)
    fit_TV_map, entropy = TV_filter(fit_map, l, fit_map.grid.shape, cell, space_group)
    mtz_TV              = dsutils.from_gemmi(io.map2mtz(fit_TV_map, highres))
    mtz_TV              = mtz_TV[mtz_TV.index.isin(mtz.index)]
    F_plus              = np.array(mtz_TV['FWT'].astype("float"))
    phi_plus            = np.radians(np.array(mtz_TV['PHWT'].astype("float")))

    # Call to function that does projection
    new_amps, new_phases, proj_error, z = TV_projection(np.array(mtz[Foff]).astype("float"), np.array(mtz[Fon]).astype("float"), np.radians(np.array(mtz[phicalc]).astype("float")), F_plus, phi_plus, ws)
    test_proj_error = proj_error[np.array(flags).astype(bool)]
    phase_changes   = np.abs(np.array(mtz["phases-pos"]-new_phases))
    phase_changes   = dsutils.adjust_phi_interval(phase_changes)
    phase_change    = np.mean(phase_changes)

    return new_amps, new_phases, test_proj_error, entropy, phase_change, z


def TV_projection(Foff, Fon, phi_calc, F_plus, phi_plus, ws):

    """
    Project a TV denoised difference structure factor vector onto measured Fon magnitude 
    to obtain new phases estimates for the difference structure factors.

    Parameters :

    1. Foff, Fon                      : (1D arrays) of measured amplitudes used for the 'Fon - Foff' difference
    2. phi_calc                       : (1D array)  of phases calculated from refined 'off' model
    3. F_plus, phi_plus               : (1D arrays) from TV denoised 'Fon - Foff' electron density map
    
    Returns :
    
    1. new amps, new phases           : (1D array) for new difference amplitude and phase estimates after the iteration 
    2. proj_error                     : (1D array) magnitude of the projection of the TV difference vector onto Fon

    """
    
    z           =   F_plus*np.exp(phi_plus*1j) + Foff*np.exp(phi_calc*1j)
    p_deltaF    =   (Fon / np.absolute(z)) * z - Foff*np.exp(phi_calc*1j)
    
    new_amps   = np.absolute(p_deltaF).astype(np.float32) * ws
    new_phases = np.angle(p_deltaF, deg=True).astype(np.float32)

    proj_error = np.absolute(np.absolute(z) - Fon)
    
    return new_amps, new_phases, proj_error, z


def find_TVmap(mtz, Flabel, philabel, name, path, map_res, cell, space_group, percent=0.07, flags=None, highres=None):

    """
    Find two TV denoised maps (one that maximizes negentropy and one that minimizes free set error) from an initial mtz and associated information.
    
    Optional arguments are whether to generate a new set of free (test) reflections – and specify what percentage of total reflections to reserve for this -
    and whether to apply a resolution cutoff.
    Screen the regularization parameter (lambda) from 0 to 0.1

    Required Parameters :

    1. MTZ, Flabel, philabel : (rsDataset) with specified structure factor and phase labels (str)
    2. name, path            : (str) for test/fit set and MTZ output
    3. map_res               : (float) spacing for map generation
    4. cell, space group     : (array) and (str)

    Returns :

    1. The two optimal TV denoised maps (GEMMI objects) and corresponding regularization parameter used (float)
    2. Errors and negentropy values from the regularization screening (numpy arrays)
    3. Mean changes in amplitude and phase between TV-denoised and observed datasets

    """

    mtz_pos = dsutils.positive_Fs(mtz, philabel, Flabel, "ogPhis_pos", "ogFs_pos")

    #if Rfree flags were specified, use these
    if flags is not None:
        test_set, fit_set, choose_test = validate.make_test_set(mtz_pos, percent, "ogFs_pos", name, path, flags) #choose_test is Boolean array to select free (test) reflections from entire set
    
    #Keep 3% of reflections for test set
    else:
        test_set, fit_set, choose_test = validate.make_test_set(mtz_pos, percent, "ogFs_pos", name, path)

    #Loop through values of lambda
    
    print('Scanning TV weights')
    lambdas       = np.linspace(1e-8, 0.1, 200)
    errors        = []
    entropies     = []
    amp_changes   = []
    phase_changes = []
    #phase_corrs   = []

    for l in tqdm(lambdas):
        fit_map             = dsutils.map_from_Fs(mtz_pos, "fit-set", "ogPhis_pos", map_res)
        fit_TV_map, entropy = TV_filter(fit_map, l, fit_map.grid.shape, cell, space_group)
            
        if highres is not None:
            Fs_fit_TV       = dsutils.from_gemmi(io.map2mtz(fit_TV_map, highres))
        else:
            Fs_fit_TV       = dsutils.from_gemmi(io.map2mtz(fit_TV_map, np.min(mtz_pos.compute_dHKL()["dHKL"])))
        
        Fs_fit_TV           = Fs_fit_TV[Fs_fit_TV.index.isin(mtz_pos.index)]
        test_TV             = Fs_fit_TV['FWT'][choose_test]
        amp_change          = np.array(mtz_pos["ogFs_pos"]) - np.array(Fs_fit_TV["FWT"])
        phase_change        = np.abs(np.array(mtz_pos["ogPhis_pos"]) - np.array(Fs_fit_TV["PHWT"]))
        #phase_corr          = np.abs(np.array(Fs_fit_TV["PHWT"]) - positive_Fs(mtz, "light-phis", Flabel, "lightPhis_pos", "ogFs_pos")["lightPhis_pos"])

        phase_change        = dsutils.adjust_phi_interval(phase_change)
        error               = np.sum(np.array(test_set) - np.array(test_TV)) ** 2
        errors.append(error)
        entropies.append(entropy)
        amp_changes.append(amp_change)
        #phase_corrs.append(phase_corr)
        phase_changes.append(phase_change)
    
    #Normalize errors
    errors    = np.array(errors)/len(errors)
    entropies = np.array(entropies)

    #Find lambda that minimizes error and that maximizes negentropy
    lambda_best_err       = lambdas[np.argmin(errors)]
    lambda_best_entr      = lambdas[np.argmax(entropies)]
    TVmap_best_err,  _    = TV_filter(fit_map, lambda_best_err,  fit_map.grid.shape, cell, space_group)
    TVmap_best_entr, _    = TV_filter(fit_map, lambda_best_entr, fit_map.grid.shape, cell, space_group)
    
    return TVmap_best_err, TVmap_best_entr, lambda_best_err, lambda_best_entr, errors, entropies, amp_changes, phase_changes #, phase_corrs