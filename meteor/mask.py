
import numpy as np
import gemmi as gm

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