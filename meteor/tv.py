import numpy as np
from   tqdm import tqdm
import pandas as pd
from   skimage.restoration import denoise_tv_chambolle

from . import maps
from . import validate
from . import dsutils
from . import io
from . import classes

def TV_filter(map, l_value, grid_size, cell, space_group):
    """
        Apply TV filtering to a Gemmi map and compute negentropy for the denoised array.

        Args:
            map (gemmi.Map): The Gemmi map object to be filtered.
            l_value (float): The regularization parameter for filtering (lambda value).
            grid_size (list): The dimensions of the grid for the map.
            cell (list): The unit cell parameters.
            space_group (str): The space group.

        Returns:
            tuple: A tuple containing the denoised map (Gemmi map object) and 
            the associated negentropy (float).
        """
    TV_arr     = denoise_tv_chambolle(np.array(map.grid), 
                                      eps=0.00000005, 
                                      weight=l_value, 
                                      max_num_iter=50)
    
    entropy    = validate.negentropy(TV_arr.flatten())
    TV_map     = maps.make_map(TV_arr, grid_size, cell, space_group)

    return TV_map, entropy

def apply_TV(itTV_mtz, pdb_params):
    """
    Project a TV denoised difference structure factor vector onto measured Fon magnitude
    to obtain new phase estimates for the difference structure factors.

    Args:
        itTV_mtz (itTVData) : Object containing original Fon and Foff data
        pdb_params (DataInfo): Object specifying dataset parameters
                               e.g. SpaceGroup, Cell, MapRes ,HighRes

    Returns:
        F_plus (ndarray)    : 1D array from TV denoised 'Fon - Foff' DED map
        phi_plus (ndarray)  : 1D array of phase values from 
                              TV denoised 'Fon - Foff' DED map
    """
    
    map_data        = dsutils.map_from_Fs(itTV_mtz.Data.itDiff, 
                                      itTV_mtz.Labels.FDiff, 
                                      itTV_mtz.Labels.PhiDiff, 
                                      pdb_params.Maps.MapRes)
    
    TV_map, entropy     = TV_filter(map_data, 
                                    itTV_mtz.Meta.lvalue, 
                                    map_data.grid.shape, 
                                    pdb_params.Xtal.Cell, 
                                    pdb_params.Xtal.SpaceGroup)
    
    mtz_TV              = dsutils.from_gemmi(io.map_to_mtz(TV_map, 
                                            pdb_params.Maps.HighRes))
    mtz_TV              = mtz_TV[mtz_TV.index.isin(itTV_mtz.Data.RawData.index)]

    F_plus              = np.array(mtz_TV['FWT'].astype("float"))
    phi_plus            = np.radians(np.array(mtz_TV['PHWT'].astype("float")))

    # Assertions
    assert len(F_plus) == len(phi_plus), "Mismatch in lengths of F_plus and phi_plus"
    assert len(F_plus) == len(phi_plus) == len(itTV_mtz.Data.RawData), "Mismatch in \
        lengths of F_plus, phi_plus, and itTV_mtz.Data.RawData"

    return entropy, F_plus, phi_plus

def TV_iteration(itTV_mtz, F_plus, phi_plus, pdb_params):
    """
    Perform one iteration of itTV denoising and return the latest 
    estimate for difference structure factors.

    Args:
        itTV_mtz (itTVData)     : An instance of itTVData containing input data.
        F_plus   (np.ndarray)   : TV denoised structure factors.
        phi_plus (np.ndarray)   : TV denoised phases.

    Returns:
        itTVData: An instance of itTVData with updated difference 
                  structure factors and iteration statistics.
    """


    # Initialize new itTVData object 

    out_mtz            = classes.itTVData()

    # Keep values that remain unchanged from itTV_mtz
    out_mtz.Meta.__dict__.update(vars(itTV_mtz.Meta))
    out_mtz.Data.RawData    = itTV_mtz.Data.RawData
    out_mtz.itStats.entropy = itTV_mtz.itStats.entropy    
    
    # Copy label values from itTV_mtz
    label_attrs = ["Fon", "Foff", "PhiC"]
    for attr in label_attrs:
        setattr(out_mtz.Labels, attr, getattr(itTV_mtz.Labels, attr))

    # Call to function that does projection
    new_amps, new_phis, out_mtz.itStats.deltaproj, out_mtz.itStats.zvec = TV_projection(
                                                                            itTV_mtz, 
                                                                            F_plus, 
                                                                            phi_plus)
    
    old_phis        =   np.array(
                        itTV_mtz.Data.itDiff[itTV_mtz.Labels.PhiDiff]).astype(np.float32)
    phase_changes   = dsutils.adjust_phi_interval(
                                np.abs(old_phis - new_phis)
                                    )
    out_mtz.itStats.deltaphi    = np.mean(phase_changes)

    # Fill out new estimate entries in out_mtz
    out_mtz.Labels.FDiff   = "new_amps"
    out_mtz.Labels.PhiDiff = "new_phases"

    out_mtz.Data.itDiff = dsutils.from_dataframe( 
                                    pd.DataFrame(
                                    {"new_amps": new_amps, "new_phases": new_phis}),
                                    pdb_params,
                                    itTV_mtz.Data.itDiff.index,
                                    ["SFAmplitude", "Phase"])

    return out_mtz


def TV_projection(itTV_mtz, F_plus, phi_plus):

    """
    Project a TV denoised difference structure factor vector onto measured Fon magnitude
    to obtain new phases estimates for the difference structure factors.

    Args :

        itTVmtz (itTVData)   : contaning original Fon and Foff data
        F_plus, phi_plus     : (1D arrays) from TV denoised 'Fon - Foff' DED map
    
    Returns :
    
        new amps, new phases : (1D array) for new difference amplitude 
                              and phase estimates after the iteration 
        delta_proj           : (1D array) magnitude of the projection of 
                              the TV difference vector onto Fon
        new light phase estimate (float)

    """

    phic     = np.radians(
                np.array(
                itTV_mtz.Data.RawData[itTV_mtz.Labels.PhiC]).astype(np.float32))
    Fon      = np.array(itTV_mtz.Data.RawData[itTV_mtz.Labels.Fon ]).astype(np.float32)
    Foff     = np.array(itTV_mtz.Data.RawData[itTV_mtz.Labels.Foff]).astype(np.float32)

    exponent   =   np.exp( phic * 1j )
    z          =   F_plus*np.exp( phi_plus*1j ) + Foff * exponent
    p_deltaF   =   ( Fon / np.absolute(z) ) * z - Foff * exponent
    
    new_amps   = np.absolute(  p_deltaF           ).astype(np.float32)
    new_phases = np.angle(     p_deltaF, deg=True ).astype(np.float32)
    delta_proj = np.absolute( np.absolute(z) - Fon )

    return new_amps, new_phases, delta_proj, np.angle(z, deg=True)


def generate_TVmaps(mtz, F_lab, phi_lab, pdb_params, CV_flags):

    """
        Generates TV maps based on the input parameters.

    Args:
        mtz (rsDataset)  : Input mtz 
        F_lab (str)      : Label for F values in mtz.
        phi_lab (str)    : Label for phi values in mtz.
        pdb_params (DataInfo): Object specifying dataset parameters
                               e.g. SpaceGroup, Cell, MapRes ,HighRes
        CV_flags (numpy.ndarray): test set flags, boolean

    Returns:
        TV_maps (list): TV maps, each is a TVMap Class object.
    """

    #ensure positive amplitudes
    mtz_pos = dsutils.positive_Fs(mtz, phi_lab, F_lab)
    assert (mtz_pos[F_lab + "_pos"] >= 0).all(), "Negative F values found in mtz_pos"
        
    split_set = validate.make_test_set(mtz_pos, CV_flags)

    #Loop through values of lambda with screen lambda function
    print('Scanning TV weights')
    TV_maps = screen_lambda(split_set, F_lab+"_pos", 
                            phi_lab+"_pos", pdb_params)
    
    return TV_maps

def mark_best_TVmap(maps):

    """
    Marks the best TV map in a list of TVMap objects based on entropy and CV error.
    This is done by updating the Meta attributes of the best maps to "NEGE" or "CV"
    (respectively).
    """
    # Get the index of the TV_map with the highest "Entropy" value
    max_entropy_idx = max(range(len(maps)), key=lambda i: maps[i].Stats.Entropy)
    max_entropy_map = maps[max_entropy_idx]
    setattr(max_entropy_map.Meta, "BestType", "NEGE")

    # Get the index of the TV_map with the lowest "CVError" value
    min_cverror_idx = min(range(len(maps)), key=lambda i: maps[i].Stats.CVError)
    min_cverror_map = maps[min_cverror_idx]
    setattr(min_cverror_map.Meta, "BestType", "CV")

    return maps

def screen_lambda(mtzs, F_lab, phi_lab, pdb_params):

    """
    Applies TV filtering to a series of maps with varying lambda values.

    Args:
        mtzs (list)          : List of rsDataset objects containing structure factors.
        F_lab (array-like)   : Experimental structure factor amplitudes.
        phi_lab (array-like) : Experimental structure factor phases.
        pdb_params (DataInfo): Object specifying dataset parameters
                               e.g. SpaceGroup, Cell, MapRes
        h_res (float or None): High resolution limit.

    Returns:
        list: List of TVMap objects with TV-filtered maps at different lambda values.
    """

    lambdas     = np.linspace(1e-8, 0.1, 200)
    TV_maps     = []

    #TO DO : Raise error is required pdb_params entries are empty

    for l_val in tqdm(lambdas):
        fit_map             = dsutils.map_from_Fs(mtzs[1], F_lab, phi_lab, 
                                        pdb_params.Maps.MapRes)
                                            
        fit_TV_map, entropy = TV_filter(fit_map, l_val, fit_map.grid.shape, 
                                        pdb_params.Xtal.Cell, 
                                        pdb_params.Xtal.SpaceGroup)
          
        Fs_TV       = dsutils.from_gemmi(io.map_to_mtz(fit_TV_map, 
                                                    pdb_params.Maps.HighRes))
     
        TV_map   = classes.TVMap()
        setattr(TV_map.Stats, "Entropy", entropy)
        setattr(TV_map.Data,  "MapData", fit_TV_map)
        setattr(TV_map.Meta,  "Lambda" , l_val)
                
        #call lambda_screen_stats
        TV_stats = lambda_screen_stats(mtzs[1], mtzs[0], Fs_TV, F_lab, phi_lab)
        label_attrs = ["CVError", "DeltaF", "DeltaPhi"]
        for idx, attr in enumerate(label_attrs):
                setattr(TV_map.Stats, attr, TV_stats[idx])
        
        TV_maps.append(TV_map)

    return TV_maps
    
def lambda_screen_stats(fit, test, tved, F_lab, phi_lab):

    """Calculates statistics for TV lambda screening.

    Args:
        fit (pandas.DataFrame) : Dataframe containing cross-validation fit data.
        test (pandas.DataFrame): Dataframe containing cross-validation test data.
        tved (pandas.DataFrame): Dataframe containing TVED data.
        F_lab (str): Label for amplitude data.
        phi_lab (str): Label for phase data.

    Returns:
        tuple: A tuple containing three elements:
            - CV_error (float):      Cross-validation error.
            - F_change (numpy.ndarray)  : Array of amplitude changes.
            - phi_change (numpy.ndarray): Array of phi changes.
    """

    #TO DO: remove hard-coded FWT and PHWT, though internal?
        
    fit_TV          = tved[tved.index.isin(fit.index)]    
    test_TV         = tved[tved.index.isin(test.index)]
    
    F_change        = np.array(fit[F_lab]) - np.array(fit_TV["FWT"]) 
    CV_error        = np.sum(np.array(test[F_lab]) - np.array(test_TV["FWT"])) ** 2
    
    phi_change      = np.abs(np.array(fit[phi_lab]) - np.array(fit_TV["PHWT"]))
    phi_change      = dsutils.adjust_phi_interval(phi_change)

    return CV_error, F_change, phi_change

