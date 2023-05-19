
import numpy as np
from   scipy.stats import differential_entropy

from . import mask

def negentropy(X):
    """
    Return negentropy (float) of X (numpy array)

    negetropy is the difference between the entropy of samples x
    # and a Gaussian with same variance
    # http://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/

    Raises:
        ValueError: If X is empty or has a single element.
    """

    if len(X) == 0:
        raise ValueError("Input array X is empty.")
    elif len(X) == 1:
        raise ValueError("Input array X has a single element.")

    std = np.std(X)
    neg_e = 0.5 * np.log(2.0 * np.pi * std ** 2) + 0.5 - differential_entropy(X)
    
    assert neg_e >= 0.0, "Negentropy is negative : %.12f." % (neg_e)

    return neg_e

def generate_CV_flags(array_length, percent=0.03):
    np.random.seed(42)
    flags = np.random.binomial(1, percent, array_length).astype(bool)
    return flags

def make_test_set(df, flags):
    """
    Divide data from an MTZ in a "fit" set and a "test" set.
    
    Additionally save test set indices as numpy object.

    Args:
        df (rs.Dataset) : The dataset to split.
        Fs (str)        : Label for structure factors to split.
        flags (np.ndarray, Bool): An array of flags to specify 
                                    which data to include in the fit set.

    Returns:
        rs.Dataset: The test set.
        rs.Dataset: The fit set.
    """
    test_set = df[flags]  # e.g. 3%
    fit_set  = df[~flags] # 97%

    test_percent = len(test_set) / len(df) * 100
    fit_percent = len(fit_set) / len(df) * 100

    #raise an error if test_percent > fit_percent
    assert test_percent <= fit_percent, "Test set percentage exceeds fit set percentage"

    return test_set, fit_set



def get_corrdiff(map1, map2, center, radius, pdb_params):
    """
    Function to find the correlation coefficient difference 
    between two maps in local and global regions.
    
    FIRST applies solvent mask to the maps map.
    THEN applies a mask around a specified region
    
    Parameters:
    map1, map2      : (GEMMI objects) to be compared
    center          : (numpy array) XYZ coordinates in PDB for the region of interest
    radius          : (float) radius for local region of interest
    pdb_params (DataInfo): Object specifying dataset parameters
                           e.g. SpaceGroup, Cell, MapRes
    
    Returns:
    diff            : (float) difference between local and global
                        CC 'map2'-'map1' values
    CC_loc, CC_glob : (numpy array) local and global CC of 'map2'-'map1' values
    """
     
    map2_nosolvent = mask.solvent_mask(np.array(map2.grid), pdb_params)
    map1_nosolvent = mask.solvent_mask(np.array(map1.grid), pdb_params)
    reg_mask = mask.get_mapmask(map2.grid, center, radius)
    loc_reg  = np.array(reg_mask, copy=True).ravel().astype(bool)

    CC_loc  = np.corrcoef(np.array(map2.grid).ravel()[loc_reg], 
                          np.array(map1.grid).ravel()[loc_reg])[0, 1]
    CC_glob = np.corrcoef(map2_nosolvent[~loc_reg].ravel(), 
                          map1_nosolvent[~loc_reg].ravel())[0, 1]

    diff = CC_glob - CC_loc

    return diff, CC_loc, CC_glob





