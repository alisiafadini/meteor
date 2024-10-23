from pathlib import Path

import gemmi

from .rsmap import Map


def sf_calc(structure: gemmi.Structure, *, high_resolution_limit: float) -> Map:
    density_map = gemmi.DensityCalculatorX()
    density_map.d_min = high_resolution_limit
    density_map.grid.setup_from(structure)
    for i, _ in enumerate(structure):
        density_map.put_model_density_on_grid(structure[i])

    ccp4_map = gemmi.Ccp4Map()
    ccp4_map.grid = density_map.grid
    ccp4_map.update_ccp4_header()

    return Map.from_ccp4_map(ccp4_map, high_resolution_limit=high_resolution_limit)


def pdb_to_calculated_map(pdb_file: Path, *, high_resolution_limit: float) -> Map:
    if not pdb_file.exists():
        msg = f"could not find file: {pdb_file}"
        raise OSError(msg)
    structure = gemmi.read_structure(str(pdb_file))
    return sf_calc(structure, high_resolution_limit=high_resolution_limit)
