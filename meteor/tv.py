import numpy as np
from tqdm import tqdm
from skimage.restoration import denoise_tv_chambolle


def tv_denoise_single_pass(mtz):
    ...


def find_TV_reg_lambda(
    mtz,
    Flabel,

):
    
    print("Scanning TV weights")
    lambdas = np.linspace(1e-8, 0.4, 100)
    
    entropies = []
    amp_changes = []
    phase_changes = []

    for l in tqdm(lambdas):
        fit_map = dsutils.map_from_Fs(mtz_pos, "fit-set", "ogPhis_pos", map_res)
        fit_TV_map, entropy = TV_filter(
            fit_map, l, fit_map.grid.shape, cell, space_group
        )
        Fs_TV = dsutils.from_gemmi(io.map2mtz(fit_TV_map, highres))
        
        entropies.append(entropy)
        amp_changes.append(amp_change)
        phase_changes.append(phase_change)
    
    entropies = np.array(entropies)
    best_entropy = np.max(entropies)
    lambda_best_entr = lambdas[np.argmax(entropies)]
    print(f"BEST ENTROPY VALUE : {best_entropy} ")
    print(f"ASSOCIATED LAMBDA VALUE : {lambda_best_entr} ")

    TVmap_best_entr, _ = TV_filter(
        fit_map, lambda_best_entr, fit_map.grid.shape, cell, space_group
    )
    
    return TVmap_best_entr, 




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

    

    TV_arr = denoise_tv_chambolle(
        np.array(map.grid), eps=0.00000005, weight=l, max_num_iter=50
    )
    entropy = validate.negentropy(TV_arr.flatten())
    TV_map = maps.make_map(TV_arr, grid_size, cell, space_group)

    return TV_map, entropy


def TV_iteration(
    mtz,
    diffs,
    phi_diffs,
    Fon,
    Foff,
    phicalc,
    map_res,
    cell,
    space_group,
    flags,
    l,
    highres,
    ws,
):
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

    z = F_plus * np.exp(phi_plus * 1j) + Foff * np.exp(phi_calc * 1j)
    p_deltaF = (Fon / np.absolute(z)) * z - Foff * np.exp(phi_calc * 1j)

    # new_amps   = np.absolute(p_deltaF).astype(np.float32) * ws
    new_amps = np.absolute(p_deltaF).astype(np.float32)
    new_phases = np.angle(p_deltaF, deg=True).astype(np.float32)

    proj_error = np.absolute(np.absolute(z) - Fon)

    return new_amps, new_phases, proj_error, np.angle(z, deg=True)


