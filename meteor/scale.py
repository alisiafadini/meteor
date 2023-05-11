
import numpy as np
import scipy.optimize as opt

def scale_iso(data1, data2, ds):

    """
    Isotropic resolution-dependent scaling of data2 to data1.
    (minimize [dataset1 - c*exp(-B*sintheta**2/lambda**2)*dataset2]

    Input :

    1. dataset1 in form of 1D numpy array
    2. dataset2 in form of 1D numpy array
    3. dHKLs for the datasets in form of 1D numpy array

    Returns :

    1. entire results from least squares fitting
    2. c (as float)
    3. B (as float)
    2. scaled dataset2 in the form of a 1D numpy array

    """
        
    def scale_func(p, x1, x2, qs):
        return x1 - (p[0]*np.exp(-p[1]*(qs**2)))*x2
    
    p0 = np.array([1.0, -20])
    qs = 1/(2*ds)
    matrix = opt.least_squares(scale_func, p0, args=(data1, data2, qs))
    
    return matrix.x[0], matrix.x[1], (matrix.x[0]*np.exp(-matrix.x[1]*(qs**2)))*data2


def scale_aniso(x_dataset, y_dataset, Miller_indx):

    """"
    Author: Virginia Apostolopoulou
    Anisotropically scales y_dataset to x_dataset given an ndarray of Miller indices.
    
    Currently only scaling function implemented in METEOR
    """

    p0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    matrix_ani = opt.least_squares(aniso_scale_func, p0, args=(x_dataset, y_dataset, Miller_indx))

    h = Miller_indx[:,0]
    k = Miller_indx[:,1]
    l = Miller_indx[:,2]
    h_sq = np.square(h)
    k_sq = np.square(k)
    l_sq = np.square(l)

    hk_prod = h*k
    hl_prod = h*l
    kl_prod = k*l

    t = - (h_sq * matrix_ani.x[1] + k_sq * matrix_ani.x[2] + l_sq * matrix_ani.x[3] 
       + 2*hk_prod * matrix_ani.x[4] + 2*hl_prod * matrix_ani.x[5] + 2*kl_prod * matrix_ani.x[6])

    data_ani_scaled = (matrix_ani.x[0]*np.exp(t))*y_dataset
    
    return matrix_ani, t,  data_ani_scaled


def aniso_scale_func(p, x1, x2, H_arr):

    "Author: Virginia Apostolopoulou"
    
    h = H_arr[:,0]
    k = H_arr[:,1]
    l = H_arr[:,2]
    
    h_sq = np.square(h)
    k_sq = np.square(k)
    l_sq = np.square(l)
    
    hk_prod = h*k
    hl_prod = h*l
    kl_prod = k*l

    t = - (h_sq * p[1] + k_sq * p[2] + l_sq * p[3] + 
        2*hk_prod * p[4] + 2*hl_prod * p[5] + 2*kl_prod * p[6])    
    expnt = np.exp( t )
    r = x1 - p[0] * expnt * x2  
    return r