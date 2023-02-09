import numpy                as np
import reciprocalspaceship  as rs
import scipy.optimize       as opt
import gemmi                as gm
import pandas               as pd
from   tqdm                 import tqdm
from   biopandas.pdb        import PandasPdb
from   scipy.stats          import differential_entropy, binned_statistic
from   skimage.restoration  import denoise_tv_chambolle

import os #remove if possible

def adjust_phi_interval(phi):

    """ Given a set of phases, return the equivalent in -180 <= phi <= 180 interval"""

    phi = phi%360
    phi[phi > 180] -=360
        
    assert np.min(phi) >= -181
    assert np.max(phi) <= 181

    return phi


def get_mapmask(grid, position, r) :
    
    """
    Returns mask (numpy array) of a map (GEMMI grid element) : mask radius 'r' (float) with center at 'position' (numpy array)
    """
    grid
    grid.fill(0)
    grid.set_points_around(gm.Position(position[0], position[1], position[2]), radius=r, value=1)
    grid.symmetrize_max()
    
    return np.array(grid, copy=True)

def solvent_mask(pdb,  cell, map_array, spacing) :

    """
    Applies a solvent mask to an electron density map

    Parameters :

    pdb, cell    : (str) and (list) PDB file name and unit cell information
    map_array    : (numpy array) of map to be masked
    spacing      : (float) spacing to generate solvent mask

    Returns :

    Flattened (1D) numpy array of mask

    """
    
    st = gm.read_structure(pdb)
    solventmask = gm.FloatGrid()
    solventmask.setup_from(st, spacing=spacing)
    solventmask.set_unit_cell(gm.UnitCell(cell[0], cell[1], cell[2], cell[3], cell[4], cell[5]))
    
    masker         = gm.SolventMasker(gm.AtomicRadiiSet.Constant, 1.5)
    masker.rprobe  = 0.9
    masker.rshrink = 1.1

    masker.put_mask_on_float_grid(solventmask, st[0])
    nosolvent = np.where(np.array(solventmask)==0, map_array, 0)
    
    return nosolvent.flatten()

def subset_to_FSigF(mtzpath, data_col, sig_col, column_names_dict={}):
    """
    Utility function for reading MTZ and returning DataSet with F and SigF.
    
    Parameters
    ----------
    mtzpath : str, filename
        Path to MTZ file to read
    data_col : str, column name
        Column name for data column. If Intensity is specified, it will be
        French-Wilson'd.
    sig_col : str, column name
        Column name for sigma column. Must select for a StandardDeviationDtype.
    column_names_dict : dictionary
        If particular column names are desired for the output, this can be specified
        as a dictionary that includes `data_col` and `sig_col` as keys and what
        values they should map to.
        
    Returns
    -------
    rs.DataSet
    """
    mtz = rs.read_mtz(mtzpath)

    # Check dtypes
    if not isinstance(
        mtz[data_col].dtype, (rs.StructureFactorAmplitudeDtype, rs.IntensityDtype)
    ):
        raise ValueError(
            f"{data_col} must specify an intensity or |F| column in {mtzpath}"
        )
    if not isinstance(mtz[sig_col].dtype, rs.StandardDeviationDtype):
        raise ValueError(
            f"{sig_col} must specify a standard deviation column in {mtzpath}"
        )

    # Run French-Wilson if intensities are provided
    if isinstance(mtz[data_col].dtype, rs.IntensityDtype):
        scaled = rs.algorithms.scale_merged_intensities(
            mtz, data_col, sig_col, mean_intensity_method="anisotropic"
        )
        mtz = scaled.loc[:, ["FW-F", "FW-SIGF"]]
        mtz.rename(columns={"FW-F": data_col, "FW-SIGF": sig_col}, inplace=True)
    else:
        mtz = mtz.loc[:, [data_col, sig_col]]

    mtz.rename(columns=column_names_dict, inplace=True)
    return mtz

def subset_to_FandPhi(mtzpath, data_col, phi_col, column_names_dict={}, flags_col=None):
    
    """
    Utility function for reading MTZ and returning DataSet with F and Phi.
    
    Parameters
    ----------
    mtzpath : str, filename
        Path to MTZ file to read
    data_col : str, column name
        Column name for data column.
    phi_col : str, column name
        Column name for phase column. Must select for a PhaseDtype.
    column_names_dict : dictionary
        If particular column names are desired for the output, this can be specified
        as a dictionary that includes `data_col` and `phi_col` as keys and what
        values they should map to.
        
    Returns
    -------
    rs.DataSet
    
    """
    
    mtz = rs.read_mtz(mtzpath)

    # Check dtypes
    if not isinstance(
        mtz[data_col].dtype, (rs.StructureFactorAmplitudeDtype)
    ):
        raise ValueError(
            f"{data_col} must specify an |F| column in {mtzpath}"
        )
    if not isinstance(mtz[phi_col].dtype, rs.PhaseDtype):
        raise ValueError(
            f"{phi_col} must specify a phase column in {mtzpath}"
        )
    
    if flags_col is not None:
        mtz = mtz.loc[:, [data_col, phi_col, flags_col]]
    else:
        mtz = mtz.loc[:, [data_col, phi_col]]

    mtz.rename(columns=column_names_dict, inplace=True)
    return mtz


def get_pdbinfo(pdb):

    """
    From a PDB file path (str), return unit cell and space group information.
    """

    pdb         = PandasPdb().read_pdb(pdb)
    text        = '\n\n%s\n' % pdb.pdb_text[:]
    info        = ['{}'.format(line) for line in text.split("\n") if line.startswith('CRYST1')]
    unit_cell   = [float(i) for i in info[0].split()[1:7]]
    space_group = ''.join(info[0].split()[7:11])

    return unit_cell, space_group

    
def get_Fcalcs(pdb, dmin, path):

    """
    From a PDB file path (str), calculate structure factors and return as rs.Dataset
    """

    os.system('gemmi sfcalc {pdb} --to-mtz={path}{root}_FCalcs.mtz --dmin={d}'.format(pdb=pdb, d=dmin-0.05, path=path, root=pdb.split('.')[0]))
    calcs = load_mtz('{path}{root}_FCalcs.mtz'.format(path=path, root=pdb.split('.')[0]))
    
    return calcs
   
def positive_Fs(df, phases, Fs, phases_new, Fs_new):

    """
    Convert between an MTZ format where difference structure factor amplitudes are saved as both positive and negative, to format where they are only positive.

    Parameters :

    df                 : (rs.Dataset) from MTZ of interest
    phases, Fs         : (str) labels for phases and amplitudes in original MTZ
    phases_new, Fs_new : (str) labels for phases and amplitudes in new MTZ


    Returns :

    rs.Dataset with new labels
    
    """
    
    new_phis = df[phases].copy(deep=True)
    new_Fs   = df[Fs].copy(deep=True)
    
    negs = np.where(df[Fs]<0)

    df[phases] = adjust_phi_interval(df[phases])

    for i in negs:
        new_phis.iloc[i] = df[phases].iloc[i]+180
        new_Fs.iloc[i]   = np.abs(new_Fs.iloc[i])
    
    new_phis             = adjust_phi_interval(new_phis)

    df_new               = df.copy(deep=True)
    df_new[Fs_new]       = new_Fs
    df_new[Fs_new]       = df_new[Fs_new].astype("SFAmplitude")
    df_new[phases_new]   = new_phis
    df_new[phases_new]   = df_new[phases_new].astype("Phase")
    
    return df_new

def load_mtz(mtz):

    """
    Load mtz file from path (str) and return rs.Dataset object
    """
    dataset = rs.read_mtz(mtz)
    dataset.compute_dHKL(inplace=True)
    
    return dataset

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


    
def make_map(data, grid_size, cell, space_group) :

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
    og.grid.set_unit_cell(gm.UnitCell(cell[0], cell[1], cell[2], cell[3], cell[4], cell[5]))
    og.grid.set_size(grid_size[0], grid_size[1], grid_size[2])
    og.grid.spacegroup = gm.find_spacegroup_by_name(space_group)
    og.grid.symmetrize_max()
    og.update_ccp4_header()
    
    return og
    
def map2mtzfile(map, mtz_name, high_res):
    """
    Write an MTZ file from a GEMMI map object.
    """
    sf = gm.transform_map_to_f_phi(map.grid, half_l=False)
    data = sf.prepare_asu_data(dmin=high_res-0.05, with_sys_abs=True)
    mtz = gm.Mtz(with_base=True)
    mtz.spacegroup = sf.spacegroup
    mtz.set_cell_for_all(sf.unit_cell)
    mtz.add_dataset('unknown')
    mtz.add_column('FWT', 'F')
    mtz.add_column('PHWT', 'P')
    mtz.set_data(data)
    mtz.switch_to_asu_hkl()
    mtz.write_to_file(mtz_name)
    
def map2mtz(map, high_res):
    """
    Return an rs.Dataset from a GEMMI map object.
    """
    sf = gm.transform_map_to_f_phi(map.grid, half_l=False)
    data = sf.prepare_asu_data(dmin=high_res-0.05, with_sys_abs=True)
    mtz = gm.Mtz(with_base=True)
    mtz.spacegroup = sf.spacegroup
    mtz.set_cell_for_all(sf.unit_cell)
    mtz.add_dataset('unknown')
    mtz.add_column('FWT', 'F')
    mtz.add_column('PHWT', 'P')
    mtz.set_data(data)
    mtz.switch_to_asu_hkl()

    return mtz
    
def res_cutoff(df, h_res, l_res) :
    """
    Apply specified low and high resolution cutoffs to rs.Dataset.
    """
    df = df.loc[(df['dHKL'] >= h_res) & (df['dHKL'] <= l_res)]
    return df
    
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

def map_from_mtzfile(path, Fs, phis, map_res):

    """
    Return a GEMMI CCP4 map object from a specified MTZ file path.
    
    Parameters :
    
    path     : (str) path to MTZ of interest
    Fs, phis : (str) and (str) labels for amplitudes and phases to be used
    map_res  : (float) to determine map spacing resolution
    
    """
    
    mtz  = gm.read_mtz_file('{}'.format(path))
    ccp4 = gm.Ccp4Map()
    ccp4.grid = mtz.transform_f_phi_to_map('{}'.format(Fs), '{}'.format(phis), sample_rate=map_res)
    ccp4.update_ccp4_header(2, True)
    
    return ccp4
    
def map_from_Fs(mtz, Fs, phis, map_res):
    
    """
    Return a GEMMI CCP4 map object from an rs.Dataset object
    
    Parameters :
    
    mtz      : rs.Dataset of interest
    Fs, phis : (str) and (str) labels for amplitudes and phases to be used
    map_res  : (float) to determine map spacing resolution
    
    """
    
    mtz = mtz.to_gemmi()
    ccp4 = gm.Ccp4Map()
    ccp4.grid = mtz.transform_f_phi_to_map('{}'.format(Fs), '{}'.format(phis), sample_rate=map_res)
    ccp4.update_ccp4_header(2, True)
    
    return ccp4
    
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
    entropy    = negentropy(TV_arr.flatten())
    TV_map     = make_map(TV_arr, grid_size, cell, space_group)

    return TV_map, entropy
    
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
    
    w = (1 + (sigdf**2 / (sigdf**2).mean()) + alpha*(df**2 / (df**2).mean()))
    return w**-1
    

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
    
from reciprocalspaceship import DataSet
from reciprocalspaceship.dtypes.base import MTZDtype
from reciprocalspaceship.utils import in_asu


def from_gemmi(gemmi_mtz):
    
    """
    Construct DataSet from gemmi.Mtz object
    
    If the gemmi.Mtz object contains an M/ISYM column and contains duplicated
    Miller indices, an unmerged DataSet will be constructed. The Miller indices
    will be mapped to their observed values, and a partiality flag will be
    extracted and stored as a boolean column with the label, ``PARTIAL``.
    Otherwise, a merged DataSet will be constructed.
    If columns are found with the ``MTZInt`` dtype and are labeled ``PARTIAL``
    or ``CENTRIC``, these will be interpreted as boolean flags used to
    label partial or centric reflections, respectively.
    
    Parameters
    ----------
    gemmi_mtz : gemmi.Mtz
        gemmi Mtz object
    
    Returns
    -------
    rs.DataSet
    """
    
    dataset = DataSet(spacegroup=gemmi_mtz.spacegroup, cell=gemmi_mtz.cell)

    # Build up DataSet
    for c in gemmi_mtz.columns:
        dataset[c.label] = c.array
        # Special case for CENTRIC and PARTIAL flags
        if c.type == "I" and c.label in ["CENTRIC", "PARTIAL"]:
            dataset[c.label] = dataset[c.label].astype(bool)
        else:
            dataset[c.label] = dataset[c.label].astype(c.type)
    dataset.set_index(["H", "K", "L"], inplace=True)

    # Handle unmerged DataSet. Raise ValueError if M/ISYM column is not unique
    m_isym = dataset.get_m_isym_keys()
    if m_isym and dataset.index.duplicated().any():
        if len(m_isym) == 1:
            dataset.merged = False
            dataset.hkl_to_observed(m_isym[0], inplace=True)
        else:
            raise ValueError(
                "Only a single M/ISYM column is supported for unmerged data"
            )
    else:
        dataset.merged = True

    return dataset

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

    calcs                       = get_Fcalcs(pdb, high_res, path)['FC']
    calcs                       = calcs[calcs.index.isin(mtz.index)]
    mtx_on, t_on, scaled_on     = scale_aniso(np.array(calcs), np.array(mtz[Fon]), np.array(list(mtz.index)))
    mtx_off, t_off, scaled_off  = scale_aniso(np.array(calcs), np.array(mtz[Foff]), np.array(list(mtz.index)))

    mtz["scaled_on"]   = scaled_on
    mtz["scaled_off"]  = scaled_off
    mtz["SIGF_on_s"]   = (mtx_on.x[0]  * np.exp(t_on))  * mtz[SIGon]
    mtz["SIGF_off_s"]  = (mtx_off.x[0] * np.exp(t_off)) * mtz[SIGoff] 
    
    sig_diffs          = np.sqrt(mtz["SIGF_on_s"]**2 + (mtz["SIGF_off_s"])**2)
    ws                 = compute_weights(mtz["scaled_on"] - Nbg * mtz["scaled_off"], sig_diffs, alpha=a)
    mtz["WDF"]       = ws * (mtz["scaled_on"] - mtz["scaled_off"])
    mtz["WDF"]       = mtz["WDF"].astype("SFAmplitude")
    mtz.infer_mtz_dtypes(inplace=True)

    return mtz, ws


def resolution_shells(data, dhkl, n):

    """ Average data in n resolution shells """

    mean_data   = binned_statistic(dhkl, data, statistic='mean', bins=n, range=(np.min(dhkl), np.max(dhkl)))
    bin_centers = (mean_data.bin_edges[:-1] + mean_data.bin_edges[1:]) / 2 

    return bin_centers, mean_data.statistic

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
    mtz           = positive_Fs(mtz, phi_diffs, diffs, "phases-pos", "diffs-pos")
    fit_mtz       = mtz[np.invert(flags)] 
    
    fit_map             = map_from_Fs(fit_mtz, "diffs-pos", "phases-pos", map_res)
    fit_TV_map, entropy = TV_filter(fit_map, l, fit_map.grid.shape, cell, space_group)
    mtz_TV              = from_gemmi(map2mtz(fit_TV_map, highres))
    mtz_TV              = mtz_TV[mtz_TV.index.isin(mtz.index)]
    F_plus              = np.array(mtz_TV['FWT'].astype("float"))
    phi_plus            = np.radians(np.array(mtz_TV['PHWT'].astype("float")))

    # Call to function that does projection
    new_amps, new_phases, proj_error, z = TV_projection(np.array(mtz[Foff]).astype("float"), np.array(mtz[Fon]).astype("float"), np.radians(np.array(mtz[phicalc]).astype("float")), F_plus, phi_plus, ws)
    test_proj_error = proj_error[np.array(flags).astype(bool)]
    phase_changes   = np.abs(np.array(mtz["phases-pos"]-new_phases))
    phase_changes   = adjust_phi_interval(phase_changes)
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

def scale_aniso(x_dataset, y_dataset, Miller_indx):

    """"
    Author: Virginia Apostolopoulou
    Anisotropically scales y_dataset to x_dataset given an ndarray of Miller indices.
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

    mtz_pos = positive_Fs(mtz, philabel, Flabel, "ogPhis_pos", "ogFs_pos")

    #if Rfree flags were specified, use these
    if flags is not None:
        test_set, fit_set, choose_test = make_test_set(mtz_pos, percent, "ogFs_pos", name, path, flags) #choose_test is Boolean array to select free (test) reflections from entire set
    
    #Keep 3% of reflections for test set
    else:
        test_set, fit_set, choose_test = make_test_set(mtz_pos, percent, "ogFs_pos", name, path)

    #Loop through values of lambda
    
    print('Scanning TV weights')
    lambdas       = np.linspace(1e-8, 0.1, 200)
    errors        = []
    entropies     = []
    amp_changes   = []
    phase_changes = []
    #phase_corrs   = []

    for l in tqdm(lambdas):
        fit_map             = map_from_Fs(mtz_pos, "fit-set", "ogPhis_pos", map_res)
        fit_TV_map, entropy = TV_filter(fit_map, l, fit_map.grid.shape, cell, space_group)
            
        if highres is not None:
            Fs_fit_TV       = from_gemmi(map2mtz(fit_TV_map, highres))
        else:
            Fs_fit_TV       = from_gemmi(map2mtz(fit_TV_map, np.min(mtz_pos.compute_dHKL()["dHKL"])))
        
        Fs_fit_TV           = Fs_fit_TV[Fs_fit_TV.index.isin(mtz_pos.index)]
        test_TV             = Fs_fit_TV['FWT'][choose_test]
        amp_change          = np.array(mtz_pos["ogFs_pos"]) - np.array(Fs_fit_TV["FWT"])
        phase_change        = np.abs(np.array(mtz_pos["ogPhis_pos"]) - np.array(Fs_fit_TV["PHWT"]))
        #phase_corr          = np.abs(np.array(Fs_fit_TV["PHWT"]) - positive_Fs(mtz, "light-phis", Flabel, "lightPhis_pos", "ogFs_pos")["lightPhis_pos"])

        phase_change        = adjust_phi_interval(phase_change)
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


def get_corrdiff( on_map, off_map, center, radius, pdb, cell, spacing) :

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
    on_nosolvent      = np.nan_to_num(solvent_mask(pdb, cell, on_a,  spacing))
    off_nosolvent     = np.nan_to_num(solvent_mask(pdb, cell, off_a, spacing))
    mask              = get_mapmask(on_map.grid, center, radius)
   
    loc_reg    = np.array(mask, copy=True).flatten().astype(bool)
    CC_loc     = np.corrcoef(on_a.flatten()[loc_reg], off_a.flatten()[loc_reg])[0,1]
    CC_glob    = np.corrcoef(on_nosolvent[np.logical_not(loc_reg)], off_nosolvent[np.logical_not(loc_reg)])[0,1]
    
    diff     = np.array(CC_glob) -  np.array(CC_loc)
    
    return diff, CC_loc, CC_glob
