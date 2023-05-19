
import numpy as np
import gemmi as gm

def get_mapmask(grid_in, position, r) :
    
    """
    Returns mask (numpy array) of a map (GEMMI grid element) : 
    mask radius 'r' (float) with center at 'position' (numpy array)
    """
    grid = grid_in.clone()
    grid
    grid.fill(0)
    grid.set_points_around(gm.Position(position[0], position[1], position[2]), 
                           radius=r, value=1)
    grid.symmetrize_max()
    
    return np.array(grid, copy=True)

def solvent_mask(map_array, pdb_params):
    """
    Applies a solvent mask to an electron density map.

    Args:
        map_array (numpy array): Map to be masked.
        pdb_params (DataInfo): Object specifying dataset parameters
                           e.g. SpaceGroup, Cell, MapRes


    Returns:
        numpy array: Flattened (1D) array of the mask.
    """
    pdb = pdb_params.Meta.Pdb
    cell = pdb_params.Xtal.Cell
    spacing = pdb_params.Maps.MaskSpacing

    try:
        structure = gm.read_structure(pdb)
    except Exception as e:
        raise ValueError(f"Failed to read PDB file: {e}")

    solvent_mask = gm.FloatGrid()
    solvent_mask.setup_from(structure, spacing=spacing)
    solvent_mask.set_unit_cell(gm.UnitCell(*cell))

    masker = gm.SolventMasker(gm.AtomicRadiiSet.Constant, 1.5)
    masker.rprobe = 0.9
    masker.rshrink = 1.1

    try:
        masker.put_mask_on_float_grid(solvent_mask, structure[0])
        nosolvent = np.where(np.array(solvent_mask) == 0, map_array, 0)
    except Exception as e:
        raise ValueError(f"Failed to apply solvent mask: {e}")

    return nosolvent.flatten()
