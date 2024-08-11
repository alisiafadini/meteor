import numpy as np
from scipy.stats import differential_entropy

from . import mask


def negentropy(X):
    """
    Return negentropy (float) of X (numpy array)
    """

    # negetropy is the difference between the entropy of samples x
    # and a Gaussian with same variance
    # http://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/

    std = np.std(X)
    # neg_e = np.log(std*np.sqrt(2*np.pi*np.exp(1))) - differential_entropy(X)
    neg_e = 0.5 * np.log(2.0 * np.pi * std**2) + 0.5 - differential_entropy(X)
    # assert neg_e >= 0.0 + 1e-8

    return neg_e





