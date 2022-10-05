import argparse
from   cProfile import label
from   re import I
from   tqdm import tqdm
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

    # scale second dataset ('on') and the first ('off') to FCalcs
    calcs                 = mtr.get_Fcalcs(args.refpdb[0], high_res, path)['FC']
    calcs                 = calcs[calcs.index.isin(og_mtz.index)]
    __, scaled_on         = mtr.scale_aniso(np.array(calcs), np.array(og_mtz[args.mtz[2]]), np.array(list(og_mtz.index)))
    __, scaled_off        = mtr.scale_aniso(np.array(calcs), np.array(og_mtz[args.mtz[1]]), np.array(list(og_mtz.index)))

    og_mtz["scaled_on"]   = scaled_on
    og_mtz["scaled_on"]   = og_mtz["scaled_on"].astype("SFAmplitude")
    og_mtz["scaled_off"]  = scaled_off
    og_mtz["scaled_off"]  = og_mtz["scaled_off"].astype("SFAmplitude") 
    og_mtz["diffs"]       = og_mtz["scaled_on"] - og_mtz["scaled_off"] 

    proj_errors     = []
    entropies       = []
    phase_changes   = []

    N = 300
    l = 0.005
    with tqdm(total=N) as pbar:
        for i in np.arange(N) + 1 :
            if i == 1:
                new_amps, new_phases, proj_error, entropy, phase_change = mtr.TV_iteration(og_mtz, "diffs", args.mtz[3] ,"scaled_on", "scaled_off", args.mtz[3], map_res, cell, space_group, flags, l, high_res)
                og_mtz["new_amps"]   = new_amps
                og_mtz["new_amps"]   = og_mtz["new_amps"].astype("SFAmplitude")
                og_mtz["new_phases"] = new_phases
                og_mtz["new_phases"] = og_mtz["new_phases"].astype("Phase")
                og_mtz.write_mtz("{name}_TVit{i}_{l}.mtz".format(name=name, i=i, l=l))
            else :
                new_amps, new_phases, proj_error, entropy, phase_change = mtr.TV_iteration(og_mtz, "new_amps", "new_phases" ,"scaled_on", "scaled_off", args.mtz[3], map_res, cell, space_group, flags, l, high_res)
                og_mtz["new_amps"]   = new_amps
                og_mtz["new_amps"]   = og_mtz["new_amps"].astype("SFAmplitude")
                og_mtz["new_phases"] = new_phases
                og_mtz["new_phases"] = og_mtz["new_phases"].astype("Phase")
                og_mtz.write_mtz("{name}_TVit{i}_{l}.mtz".format(name=name, i=i, l=l))
            
            # Track projection error and phase change for each iteration
            proj_errors.append(proj_error)
            entropies.append(entropy)
            phase_changes.append(phase_change)
            pbar.update()

    # Optionally plot result for errors and negentropy
    if args.plot is True:

        fig, ax1 = plt.subplots(figsize=(10,4))

        color = 'black'
        ax1.set_xlabel(r'Iteration')
        ax1.set_ylabel(r'TV Map Error (TV$_\mathrm{err}$)', color=color)
        ax1.plot(np.arange(N), proj_errors/np.max(np.array(proj_errors)), color=color, linewidth=5)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx() 

        color = 'darkgray'
        ax2.set_ylabel('Negentropy', color=color)  
        ax2.plot(np.arange(N), entropies, color=color, linewidth=5)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout() 
        plt.show()

        fig, ax1 = plt.subplots(figsize=(10,4))

        color = 'tomato'
        ax1.set_xlabel(r'Iteration')
        ax1.set_ylabel('Average Phase Difference \n (Iteration - Initial)')
        ax1.plot(np.arange(N), phase_changes, color=color, linewidth=5)
        ax1.tick_params(axis='y')

        fig.set_tight_layout(True)
        plt.show()

    print('DONE')

if __name__ == "__main__":
    main()