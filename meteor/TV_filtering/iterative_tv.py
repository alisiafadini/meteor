import argparse
from re import I
import matplotlib.pyplot as plt
import numpy as np
import os

import reciprocalspaceship as rs
import meteor.meteor  as mtr
import seaborn as sns
sns.set_context("notebook", font_scale=1.4)

"""

Iterative TV algorithm to improve phase estimates for low occupancy species.
Writes out a difference map file (MTZ) with improved phases.

"""

def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "-mtz",
        "--mtz",
        nargs=4,
        metavar=("mtz", "F_off", "F_on", "phi_col"),
        required=True,
        help=("MTZ to be used for initial map. Specified as (filename, F_off, F_on, Phi)"),
    )
    
    parser.add_argument(
        "-ref",
        "--refpdb",
        nargs=1,
        metavar=("pdb"),
        required=True,
        help=(
            "PDB to be used as reference ('off') structure. "
            "Specified as (filename)."
        ),
    )
    
    # Optional arguments
    
    parser.add_argument(
        "-d_h",
        "--highres",
        type=float,
        default=None,
        help="If set, high res to truncate maps"
    )
    
    parser.add_argument(
        "-fl",
        "--flags",
        type=str,
        default=None,
        help="If set, label for Rfree flags to use as test set"
    )
    
    parser.add_argument("--plot", help="--plot for optional plotting and saving, --no-plot to skip", action=argparse.BooleanOptionalAction)
    
    return parser.parse_args()

def main():

    # Map writing parameters
    map_res     = 4

    # Parse commandline arguments
    args = parse_arguments()
    
    path              = os.path.split(args.mtz[0])[0]
    name              = os.path.split(args.mtz[0])[1].split('.')[0]
    cell, space_group = mtr.get_pdbinfo(args.refpdb[0])
    
    print('%%%%%%%%%% ANALYZING DATASET : {n} in {p} %%%%%%%%%%%'.format(n=name, p=path))
    print('CELL         : {}'.format(cell))
    print('SPACEGROUP   : {}'.format(space_group))
    
    # Apply resolution cut if specified
    if args.highres is not None:
        high_res = args.highres
    else:
        high_res = np.min(mtr.load_mtz(args.mtz[0]).compute_dHKL()["dHKL"])
    
    # Read in mtz file
    og_mtz = mtr.load_mtz(args.mtz[0])

    # Use own R-free flags set if specified
    if args.flags is not None:
        og_mtz = og_mtz.loc[:, [args.mtz[1], args.mtz[2], args.mtz[3], args.flags]]  
        flags  = og_mtz[args.flags] == 0  

    else:
        og_mtz = og_mtz.loc[:, [args.mtz[1], args.mtz[2], args.mtz[3]]]  
        flags  = np.random.binomial(1, 0.03, og_mtz[args.mtz[1]].shape[0]).astype(bool)      

    for i in np.arange(4) + 1 :
        new_amps, new_phases, proj_error = TV_iteration(og_mtz, arg.mtz[2], args.mtz[1], args.mtz[3], name, map_res, cell, space_group, flags, 0.005, highres)

        # Track projection error and phase change for each iteration

    print('DONE')

if __name__ == "__main__":
    main()