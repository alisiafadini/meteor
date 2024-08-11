import os
import gemmi as gm
import reciprocalspaceship as rs


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
    if not isinstance(mtz[data_col].dtype, (rs.StructureFactorAmplitudeDtype)):
        raise ValueError(f"{data_col} must specify an |F| column in {mtzpath}")
    if not isinstance(mtz[phi_col].dtype, rs.PhaseDtype):
        raise ValueError(f"{phi_col} must specify a phase column in {mtzpath}")

    if flags_col is not None:
        mtz = mtz.loc[:, [data_col, phi_col, flags_col]]
    else:
        mtz = mtz.loc[:, [data_col, phi_col]]

    mtz.rename(columns=column_names_dict, inplace=True)
    return mtz


def map2mtzfile(map, mtz_name, high_res):
    """
    Write an MTZ file from a GEMMI map object.
    """

    # TODO this seems to be redundant
    #      combine with map2mtz function
    #  --> should these functions be methods on the map object?

    sf = gm.transform_map_to_f_phi(map.grid, half_l=False)
    data = sf.prepare_asu_data(dmin=high_res - 0.05, with_sys_abs=True)
    mtz = gm.Mtz(with_base=True)
    mtz.spacegroup = sf.spacegroup
    mtz.set_cell_for_all(sf.unit_cell)
    mtz.add_dataset("unknown")
    mtz.add_column("FWT", "F")
    mtz.add_column("PHWT", "P")
    mtz.set_data(data)
    mtz.switch_to_asu_hkl()
    mtz.write_to_file(mtz_name)


def map2mtz(map, high_res):
    """
    Return an rs.Dataset from a GEMMI map object.
    """
    sf = gm.transform_map_to_f_phi(map.grid, half_l=False)
    data = sf.prepare_asu_data(dmin=high_res - 0.05, with_sys_abs=True)
    mtz = gm.Mtz(with_base=True)
    mtz.spacegroup = sf.spacegroup
    mtz.set_cell_for_all(sf.unit_cell)
    mtz.add_dataset("unknown")
    mtz.add_column("FWT", "F")
    mtz.add_column("PHWT", "P")
    mtz.set_data(data)
    mtz.switch_to_asu_hkl()

    return mtz


def map_from_mtzfile(path, Fs, phis, map_res):
    """
    Return a GEMMI CCP4 map object from a specified MTZ file path.

    Parameters :

    path     : (str) path to MTZ of interest
    Fs, phis : (str) and (str) labels for amplitudes and phases to be used
    map_res  : (float) to determine map spacing resolution

    """

    mtz = gm.read_mtz_file("{}".format(path))
    ccp4 = gm.Ccp4Map()
    ccp4.grid = mtz.transform_f_phi_to_map(
        "{}".format(Fs), "{}".format(phis), sample_rate=map_res
    )
    ccp4.update_ccp4_header(2, True)

    return ccp4


def get_pdbinfo(pdb):
    """
    From a PDB file path (str), return unit cell and space group information.
    """

    # TODO clean up this function and remove dependency on biopython (!)

    # pdb         = PandasPdb().read_pdb(pdb)
    # text        = '\n\n%s\n' % pdb.pdb_text[:]

    with open(pdb, "r") as f:
        for line in f:
            if line.startswith("CRYST1"):

                split_line = line.strip().split()
                unit_cell = [float(i) for i in split_line[1:7]]
                space_group = "".join(split_line[7:11])

                return unit_cell, space_group

    # if here, we did not find the CRYST1 record
    raise IOError("could not find CRYST1 record in:", pdb)


def get_Fcalcs(pdb, dmin, path):
    """
    From a PDB file path (str), calculate structure factors and return as rs.Dataset
    """

    os.system(
        "gemmi sfcalc {pdb} --to-mtz={path}/{root}_FCalcs.mtz --dmin={d}".format(
            pdb=pdb, d=dmin - 0.05, path=path, root=pdb.split(".")[0].split("/")[-1]
        )
    )
    calcs = load_mtz(
        "{path}/{root}_FCalcs.mtz".format(
            path=path, root=pdb.split(".")[0].split("/")[-1]
        )
    )

    return calcs
