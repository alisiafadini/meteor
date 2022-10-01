import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from   tqdm import tqdm

import reciprocalspaceship as rs
import meteor.meteor  as mtr
import seaborn as sns
sns.set_context("notebook", font_scale=1.4)

"""

Apply a total variation (TV) filter to a map.
The level of filtering (determined by the regularization parameter lambda)
is chosen so as to minimize the error between denoised and measured amplitudes or maximize the map negentropy
for a free set ('test' set) of reflections.

Write two denoised map file (MTZ), one for each optimal parameter, and optionally the plot/save the plot from the lambda determination (PNG).

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
        nargs=3,
        metavar=("mtz", "F_col", "phi_col"),
        required=True,
        help=("MTZ to be used for initial map. Specified as (filename, F, Phi)"),
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

    # Use own R-free flags set if specified
    if args.flags is not None:
        og_mtz        = mtr.subset_to_FandPhi(*args.mtz, {args.mtz[1]: "F", args.mtz[2]: "Phi"}, args.flags).dropna()
        TVmap_best_err, TVmap_best_entr, lambda_best_err, lambda_best_entr, errors, entropies = mtr.find_TVmap(og_mtz, "F", "Phi", name, path, map_res, cell, space_group, flags=args.flags)
    
    else:
        # Read in mtz file
        og_mtz        = mtr.subset_to_FandPhi(*args.mtz, {args.mtz[1]: "F", args.mtz[2]: "Phi"}).dropna()
        
        # Find and save denoised maps that (1) minimizes the map error or (2) maximizes the map negentropy
        TVmap_best_err, TVmap_best_entr, lambda_best_err, lambda_best_entr, errors, entropies = mtr.find_TVmap(og_mtz, "F", "Phi", name, path, map_res, cell, space_group)
        TVFs_best_err = mtr.map2mtzfile(TVmap_best_err,  '{n}_TV_{l}_besterror.mtz'.format(n=name,   l=np.round(lambda_best_err, decimals=3)), high_res)
        TVFs_best_err = mtr.map2mtzfile(TVmap_best_entr, '{n}_TV_{l}_bestentropy.mtz'.format(n=name, l=np.round(lambda_best_entr, decimals=3)), high_res)
    
    print('Writing out TV denoised map with weights={lerr} and {lentr}'.format(lerr=np.round(lambda_best_err, decimals=3), lentr=np.round(lambda_best_entr, decimals=3)))
    
    # Optionally plot result for errors and negentropy
    if args.plot is True:
        fig, ax1 = plt.subplots(figsize=(10,5))

        color = 'black'
        ax1.set_xlabel(r'$\lambda$')
        ax1.set_ylabel(r'$\sum (\Delta \mathrm{Fobs}_\mathrm{free} - \Delta \mathrm{Fcalc}_\mathrm{free})^2$', color=color)
        ax1.plot(np.linspace(1e-8, 0.1, len(errors)), errors/np.max(errors), color=color, linewidth=5)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'silver'
        ax2.set_ylabel('Negentropy', color=color)
        ax2.plot(np.linspace(1e-8, 0.1, len(entropies)), entropies, color=color, linewidth=5)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        fig.savefig('{p}{n}lambda-optimize.png'.format(p=path, n=name))
        
        print('Saving weight scan plot')
        
    print('DONE.')


if __name__ == "__main__":
    main()


