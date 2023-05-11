import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

import seaborn as sns
sns.set_context("notebook", font_scale=1.8)

from meteor import io
from meteor import dsutils
from meteor import tv


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
    cell, space_group = io.get_pdbinfo(args.refpdb[0])
    
    print('%%%%%%%%%% ANALYZING DATASET : {n} in {p} %%%%%%%%%%%'.format(n=name, p=path))
    print('CELL         : {}'.format(cell))
    print('SPACEGROUP   : {}'.format(space_group))
    
    # Apply resolution cut if specified
    if args.highres is not None:
        high_res = args.highres
    else:
        high_res = np.min(io.load_mtz(args.mtz[0]).compute_dHKL()["dHKL"])

    # Use own R-free flags set if specified
    if args.flags is not None:
        og_mtz        = io.subset_to_FandPhi(*args.mtz, {args.mtz[1]: "F", args.mtz[2]: "Phi"}, args.flags).dropna()
        TVmap_best_err, TVmap_best_entr, lambda_best_err, lambda_best_entr, errors, entropies, amp_change, ph_change = tv.find_TVmap(og_mtz, "F", "Phi", name, path, map_res, cell, space_group, flags=args.flags)


    else:
        # Read in mtz file
        og_mtz        = io.subset_to_FandPhi(*args.mtz, {args.mtz[1]: "F", args.mtz[2]: "Phi"}).dropna()
        
        # Find and save denoised maps that (1) minimizes the map error or (2) maximizes the map negentropy
        TVmap_best_err, TVmap_best_entr, lambda_best_err, lambda_best_entr, errors, entropies, amp_change, ph_change = tv.find_TVmap(og_mtz, "F", "Phi", name, path, map_res, cell, space_group)

        io.map2mtzfile(TVmap_best_err,  '{n}_TV_{l}_besterror.mtz'.format(n=name,   l=np.round(lambda_best_err, decimals=3)), high_res)
        io.map2mtzfile(TVmap_best_entr, '{n}_TV_{l}_bestentropy.mtz'.format(n=name, l=np.round(lambda_best_entr, decimals=3)), high_res)
    
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
        ax2.set_ylabel('Negentropy', color='grey')
        ax2.plot(np.linspace(1e-8, 0.1, len(entropies)), entropies, color=color, linewidth=5)
        ax2.tick_params(axis='y', labelcolor='grey')

        fig.tight_layout()
        fig.savefig('{p}{n}lambda-optimize.png'.format(p=path, n=name))
        
        print('Saving weight scan plot')

        fig, ax = plt.subplots(figsize=(9,5))
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r' < | |F$_\mathrm{obs}$| - |F$_\mathrm{TV}$| | > ')
        ax.plot(np.linspace(1e-8, 0.1, len(entropies)), np.mean(np.abs(amp_change), axis=1), color='indigo', linewidth=5)
        fig.tight_layout()
        fig.savefig('{p}{n}F-change.png'.format(p=path, n=name))

        fig, ax = plt.subplots(figsize=(9,5))
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'< | $\phi_\mathrm{obs}$ - $\phi_\mathrm{TV}$ | > ($^\circ$)')
        ax.plot(np.linspace(1e-8, 0.1, len(entropies)), np.mean(np.abs(ph_change), axis=1), color='orangered', linewidth=5)
        ax.axhline(y=19.52, linewidth=2, linestyle='--', color='orangered')
        fig.tight_layout()
        fig.savefig('{p}{n}phi-change.png'.format(p=path, n=name))

        fig, ax = plt.subplots(figsize=(9,5))
        #ax.scatter(1/og_mtz.compute_dHKL()["dHKL"], np.abs(ph_change[np.argmin(errors)]), label="Best Error Map, mean = {}".format(np.round(np.mean(ph_change[np.argmin(errors)])), decimals=2), color='black', alpha=0.5)
        #ax.scatter(1/og_mtz.compute_dHKL()["dHKL"], np.abs(ph_change[np.argmax(entropies)]), label="Best Entropy Map, mean = {}".format(np.round(np.mean(ph_change[np.argmax(entropies)])), decimals=2), color='grey', alpha=0.5)
        
        res_mean, data_mean = dsutils.resolution_shells(np.abs(ph_change[np.argmin(errors)]), 1/og_mtz.compute_dHKL()["dHKL"], 20)
        ax.plot(res_mean, data_mean, linewidth=3, linestyle='--', color='black', label="Best Error")
        res_mean, data_mean = dsutils.resolution_shells(np.abs(ph_change[np.argmax(entropies)]), 1/og_mtz.compute_dHKL()["dHKL"], 20)
        ax.plot(res_mean, data_mean, linewidth=3, linestyle='--', color='darkgray', label="Best Entropy")
        ax.set_ylabel(r'| $\phi_\mathrm{obs}$ - $\phi_\mathrm{TV}$ | ($^\circ$)')
        ax.set_xlabel(r'1/dHKL (${\AA}^{-1}$)')
        fig.tight_layout()
        fig.legend(handlelength=0.6, loc='center right')
        fig.savefig('{p}{n}phi-change-dhkl.png'.format(p=path, n=name))

        fig, ax = plt.subplots(figsize=(9,5))
        #ax.scatter(1/og_mtz.compute_dHKL()["dHKL"], np.abs(amp_change[np.argmin(errors)]), label="Best Error Map, mean = {}".format(np.round(np.mean(amp_change[np.argmin(errors)])), decimals=2), color='black', alpha=0.5)
        #ax.scatter(1/og_mtz.compute_dHKL()["dHKL"], np.abs(amp_change[np.argmax(entropies)]), label="Best Entropy Map, mean = {}".format(np.round(np.mean(amp_change[np.argmax(entropies)])), decimals=2), color='darkgray', alpha=0.5)
        
        res_mean, data_mean = dsutils.resolution_shells(np.abs(amp_change[np.argmin(errors)]), 1/og_mtz.compute_dHKL()["dHKL"], 15)
        ax.plot(res_mean, data_mean, linewidth=3, linestyle='--', color='black', label="Best Error")
        res_mean, data_mean = dsutils.resolution_shells(np.abs(amp_change[np.argmax(entropies)]), 1/og_mtz.compute_dHKL()["dHKL"], 15)
        ax.plot(res_mean, data_mean, linewidth=3, linestyle='--', color='darkgray', label="Best Entropy")
        ax.set_ylabel(r'| |F$_\mathrm{obs}$| - |F$_\mathrm{TV}$| |')
        ax.set_xlabel(r'1/dHKL (${\AA}^{-1}$)')
        fig.tight_layout()
        fig.legend(handlelength=0.6, loc='center right')
        fig.savefig('{p}{n}amp-change-dhkl.png'.format(p=path, n=name))

        #plt.show()
        print('DONE.')


if __name__ == "__main__":
    main()


