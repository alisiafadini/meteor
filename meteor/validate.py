
import numpy as np
from   scipy.stats import differential_entropy

from . import mask

def negentropy(X):
    """
    Return negentropy (float) of X (numpy array)
    """
    
    # negetropy is the difference between the entropy of samples x
    # and a Gaussian with same variance
    # http://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/
    
    std = np.std(X)
    neg_e = np.log(std*np.sqrt(2*np.pi*np.exp(1))) - differential_entropy(X)
    #assert neg_e >= 0.0
    
    return neg_e


def make_test_set(df, percent, Fs, out_name, path, flags=False):

    """
    Write MTZ file where data from an original MTZ has been divided in a "fit" set and a "test" set.
    
    Additionally save test set indices as numpy object.

    Required parameters :

    df                : (rs.Dataset) to split
    percent           : (float) fraction of reflections to keep for test set – e.g. 0.03
    Fs                : (str) labels for structure factors to split
    out_name, path    : (str) and (str) output file name and path specifications
    
    Returns :
    
    test_set, fit_set : (rs.Dataset) and (rs.Dataset) for the two sets
    choose_test       : (1D array) containing test data indices as boolean type

    """
    if flags is not False:
        choose_test = df[flags] == 0
        
    else:
        choose_test = np.random.binomial(1, percent, df[Fs].shape[0]).astype(bool)
    test_set = df[Fs][choose_test] #e.g. 3%
    fit_set  = df[Fs][np.invert(choose_test)] #97%
    
    df["fit-set"]   = fit_set
    df["fit-set"]   = df["fit-set"].astype("SFAmplitude")
    df["test-set"]  = test_set
    df["test-set"]  = df["test-set"].astype("SFAmplitude")
    
    df.write_mtz("{path}split-{name}.mtz".format(path=path, name=out_name))
    np.save("{path}test_flags-{name}.npy".format(path=path, name=out_name), choose_test)
    
    return test_set, fit_set, choose_test


def get_corrdiff(on_map, off_map, center, radius, pdb, cell, spacing) :

    """
    Function to find the correlation coefficient difference between two maps in local and global regions.
    
    FIRST applies solvent mask to 'on' and an 'off' map.
    THEN applies a  mask around a specified region
    
    Parameters :
    
    on_map, off_map : (GEMMI objects) to be compared
    center          : (numpy array) XYZ coordinates in PDB for the region of interest
    radius          : (float) radius for local region of interest
    pdb, cell       : (str) and (list) PDB file name and cell information
    spacing         : (float) spacing to generate solvent mask
    
    Returns :
    
    diff            : (float) difference between local and global correlation coefficients of 'on'-'off' values
    CC_loc, CC_glob : (numpy array) local and global correlation coefficients of 'on'-'off' values

    """

    off_a             = np.array(off_map.grid)
    on_a              = np.array(on_map.grid)
    on_nosolvent      = np.nan_to_num(mask.solvent_mask(pdb, cell, on_a,  spacing))
    off_nosolvent     = np.nan_to_num(mask.solvent_mask(pdb, cell, off_a, spacing))
    reg_mask          = mask.get_mapmask(on_map.grid, center, radius)
   
    loc_reg    = np.array(reg_mask, copy=True).flatten().astype(bool)
    CC_loc     = np.corrcoef(on_a.flatten()[loc_reg], off_a.flatten()[loc_reg])[0,1]
    CC_glob    = np.corrcoef(on_nosolvent[np.logical_not(loc_reg)], off_nosolvent[np.logical_not(loc_reg)])[0,1]
    
    diff     = np.array(CC_glob) -  np.array(CC_loc)
    
    return diff, CC_loc, CC_glob

