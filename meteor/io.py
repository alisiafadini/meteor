import gemmi as gm
import reciprocalspaceship as rs
import os



def load_mtz(mtz):

    """
    Load mtz file from path (str) and return rs.DataSet object
    """
    dataset = rs.read_mtz(mtz)
    dataset.compute_dHKL(inplace=True)
    
    return dataset


def subset_to_FSigF(mtzpath, data_col, sig_col, column_names_dict={}):
    """
    Note: original function from reciprocalspaceship

    Utility function for reading MTZ and returning DataSet with F and SigF.

    Args:
        mtzpath (str)   : Path to the MTZ file to read.
        data_col (str)  : Column name for the data column. 
                        If "Intensity" is specified, it will be French-Wilson'd.
        sig_col (str)   : Column name for the sigma column. 
                        Must select for a StandardDeviationDtype.
        column_names_dict (dict): If particular column names are desired for the output,
                        this can be specified as a dictionary that includes `data_col` 
                        and `sig_col` as keys and what values they should map to.

    Returns:
        rs.DataSet: The constructed DataSet.
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
    Note: original function from reciprocalspaceship

    Utility function for reading MTZ and returning DataSet with F and Phi.

        Args:
            mtzpath (str)   : Path to MTZ file to read.
            data_col (str)  : Column name for data column.
            phi_col (str)   : Column name for phase column. 
            column_names_dict (dict, optional): If particular column names are 
                desired for the output, this can be specified as a dictionary 
                that includes 'data_col' and 'phi_col' as keys
                and the desired values they should map to.

        Returns:
            rs.DataSet: The constructed DataSet.
    
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


def map_to_mtz(map, high_res, mtz_name=None):
    """
    Convert a GEMMI map object to an MTZ file or rs.Dataset.

    If mtz_name is provided, an MTZ file will be written.
    If mtz_name is None, an rs.Dataset will be returned.

    Args:
        map (gemmi.Map): The GEMMI map object.
        high_res (float): The high-resolution cutoff value.
        mtz_name (str, optional): The name of the MTZ file to be written.

    Returns:
        Gemmi MTZ or None: The MTZ if mtz_name is None, otherwise None.

    """
    sf = gm.transform_map_to_f_phi(map.grid, half_l=False)
    data = sf.prepare_asu_data(dmin=high_res - 0.05, with_sys_abs=True)
    mtz = gm.Mtz(with_base=True)
    mtz.spacegroup = sf.spacegroup
    mtz.set_cell_for_all(sf.unit_cell)
    mtz.add_dataset('unknown')
    mtz.add_column('FWT', 'F')
    mtz.add_column('PHWT', 'P')
    mtz.set_data(data)
    mtz.switch_to_asu_hkl()

    if mtz_name:
        mtz.write_to_file(mtz_name)
    else:
        return mtz



def map_from_mtzfile(path, Fs, phis, map_res):

    """
    Return a GEMMI CCP4 map object from a specified MTZ file path.

    Args:
        path (str): Path to the MTZ file of interest.
        Fs (str): Label for the amplitudes to be used.
        phis (str): Label for the phases to be used.
        map_res (float): Resolution to determine map spacing.

    Returns:
        gemmi.Map: The GEMMI CCP4 map object.
    
    """
    
    mtz  = gm.read_mtz_file('{}'.format(path))
    ccp4 = gm.Ccp4Map()
    ccp4.grid = mtz.transform_f_phi_to_map('{}'.format(Fs), '{}'.format(phis), 
                                           sample_rate=map_res)
    ccp4.update_ccp4_header(2, True)
    
    return ccp4

def get_pdbinfo(pdb):
    """
    From a PDB file path, return unit cell and space group information.

    Args:
        pdb (str): Path to the PDB file.

    Returns:
        tuple: Tuple containing unit cell dimensions 
        (a, b, c, alpha, beta, gamma) and space group.

    Raises:
        IOError: If the CRYST1 record is not found in the PDB file.

    """
    with open(pdb, 'r') as f:
        for line in f:
            if line.startswith('CRYST1'):
                split_line = line.strip().split()
                unit_cell = [float(i) for i in split_line[1:7]]
                space_group = ''.join(split_line[7:11])
                return tuple(unit_cell), space_group

    raise IOError("Could not find CRYST1 record in: " + pdb)

    
#def get_Fcalcs(pdb, dmin, path):

    #os.system('gemmi sfcalc {pdb} --to-mtz={path}{root}_FCalcs.mtz 
    # --dmin={d}'.format(pdb=pdb, d=dmin-0.05, 
    # path=path, root=pdb.split('.')[0].split('/')[-1]))
    #calcs = load_mtz('{path}{root}_FCalcs.mtz'.format(path=path,
    #root=pdb.split('.')[0].split('/')[-1]))
    
    #return calcs

def get_Fcalcs(pdb, dmin):
    """
    Calculate structure factors from a PDB file and return as rs.Dataset.

    Args:
        pdb (str): Path to the PDB file.
        dmin (float): Minimum resolution for calculating structure factors.

    Returns:
        rs.Dataset: The calculated structure factors.

    Raises:
        ValueError: If the PDB file does not exist or cannot be loaded.

    """

    try:
        gm.read_pdb(pdb)
    except Exception as e:
        raise ValueError(f"Failed to load PDB file: {e}")
    
    os.system('gemmi sfcalc {pdb} --to-mtz={root}_FCalcs.mtz --dmin={d}'.format(pdb=pdb,
                                                                                d=dmin-0.05, 
                                                                                root=pdb.split('.')[0].split('/')[-1]))
    calcs = load_mtz('{root}_FCalcs.mtz'.format(root=pdb.split('.')[0].split('/')[-1]))

    return calcs
