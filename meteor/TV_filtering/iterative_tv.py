import argparse
from   tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from meteor import io
from meteor import dsutils
from meteor import maps
from meteor import tv
from meteor import classes
from meteor import validate

import seaborn as sns
sns.set_context("notebook", font_scale=1.8)

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
        nargs=6,
        metavar=("mtz", "F_off", "F_on", "phi_col", "SIGF_off", "SIGF_on"),
        required=True,
        help=("MTZ to be used for initial map. \
              Specified as (filename, F_off, F_on, Phi)"),
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
        "-a",
        "--alpha",
        type=float,
        default=0.0,
        help="alpha value for computing difference map weights (default=0.0)",
    )

    parser.add_argument(
        "-l",
        "--lambda_tv",
        type=float,
        default=0.015,
        help="lambda value for TV denoising weighting (default 0.015)",
    )
    
    parser.add_argument("--plot", help="--plot for optional plotting and saving, \
                        --no-plot to skip", action=argparse.BooleanOptionalAction)
    
    return parser.parse_args()

def main():

    # Parse commandline arguments
    args           = parse_arguments()
    
    # Fill out mtz labels of interest
    data_labels    = classes.MtzLabels()
    label_attrs = ["Foff", "Fon", "PhiC", "SigFoff", "SigFon"]
    for idx, attr in enumerate(label_attrs):
            setattr(data_labels, attr, args.mtz[idx+1])

    # Fill out the dataset information needed
    data_info                   = classes.DataInfo()
    data_info.Meta.Path         = os.path.split(args.mtz[0])[0]
    data_info.Meta.Name         = os.path.split(args.mtz[0])[1].split('.')[0]
    data_info.Maps.MapRes       = 6 #Map writing resolution
    data_info.Xtal.Cell, data_info.Xtal.SpaceGroup  = io.get_pdbinfo(args.refpdb[0])

    print('%%%%%%%%%% ANALYZING DATASET : {n} in {p} %%%%%%%%%%'.format(
                                                    n=data_info.Meta.Name, 
                                                    p=data_info.Meta.Path))
    print('CELL         : {}'.format(data_info.Xtal.Cell))
    print('SPACEGROUP   : {}'.format(data_info.Xtal.SpaceGroup))
    
    # Apply resolution cut and read in mtz data
    if args.highres is not None:
        data_info.Maps.HighRes = args.highres
    else:
        data_info.Maps.HighRes = np.min(io.load_mtz(args.mtz[0]).compute_dHKL()["dHKL"])
    
    og_mtz = dsutils.res_cutoff(io.load_mtz(args.mtz[0]), data_info.Maps.HighRes, 25)

    # scale second dataset ('on') and the first ('off') to FCalcs
    scaled_mtz = maps.scale_toFCalcs(og_mtz, data_labels, 
                                     args.refpdb[0],
                                     data_info.Maps.HighRes)
    #calculate deltaFs (weighted or not)
    diffs, _  = maps.find_w_diffs(scaled_mtz, args.alpha)


    N = 100
    print("TV Regularization, lambda = ", args.lambda_tv)

    #initiate iteration list with diffs mtz of Class itTVData
    it1_mtz   = classes.itTVData()

    # Keep values from diffs
    it1_mtz.Data.RawData    = diffs
    it1_mtz.Meta.lvalue     = args.lambda_tv

    # Ensure a start from positive amplitudes
    it1_mtz.Data.itDiff     = dsutils.positive_Fs(diffs, data_labels.PhiC, "WDF")
    it1_mtz.Labels.FDiff    = "WDF_pos"
    it1_mtz.Labels.PhiDiff  = data_labels.PhiC+"_pos"
    
    #update single inputs to scaled data
    label_attrs = ["Fon", "Foff", "PhiC"]
    new_attrs   = ["scaled_on", "scaled_off", "PHIC"]
    for idx, attr in enumerate(label_attrs):
        setattr(it1_mtz.Labels, attr, new_attrs[idx])

    it_mtzs = [it1_mtz]

    with tqdm(total=N) as pbar:
        for i in np.arange(N):

            #run apply_TV function
            entropy, Fplus, phiplus     = tv.apply_TV(it_mtzs[i], data_info)
            it_mtzs[i].itStats.entropy  = entropy 

            #run TV_iteration function
            iteration_mtz               = tv.TV_iteration(it_mtzs[i],
                                                          Fplus,
                                                          phiplus, data_info)
            #append to latest object to a list
            it_mtzs.append(iteration_mtz)
            pbar.update()

    #Save 100th 
    print("SAVING FINAL MAP")
    final_map = dsutils.map_from_Fs(it_mtzs[-1].Data.itDiff, 
                                 "new_amps", 
                                 "new_phases", 
                                 data_info.Maps.MapRes)

    
    final_map.write_ccp4_map("{name}_TVit{i}_{l}.ccp4".format(name=data_info.Meta.Name, 
                                                              i=i+1, 
                                                              l=args.lambda_tv))    

    
    # Optionally plot result for errors and negentropy
    if args.plot is True:
        
        entropies   = [itTVmtz.itStats.entropy   for itTVmtz in it_mtzs]
        deltaphis   = [itTVmtz.itStats.deltaphi  for itTVmtz in it_mtzs]
        deltaproj   = [itTVmtz.itStats.deltaproj for itTVmtz in it_mtzs]

        fig, ax1 = plt.subplots(figsize=(10,4))

        color = 'black'
        ax1.set_title('$\lambda$ = {}'.format(args.lambda_tv))
        ax1.set_xlabel(r'Iteration')
        ax1.set_ylabel(r'TV Projection Magnitude (TV$_\mathrm{proj}$)', color=color)
        ax1.plot(np.arange(N-1),  np.array(np.mean(np.array(deltaproj[1:]), axis=1)
                                           /
                                           np.max(np.mean(np.array(deltaproj[1:]), 
                                                          axis=1)))[1:], 
                                          color=color, 
                                          linewidth=5)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx() 

        color = 'silver'
        ax2.set_ylabel('Negentropy', color='grey')  
        ax2.plot(np.arange(N), entropies[1:], color=color, linewidth=5)
        ax2.tick_params(axis='y', labelcolor="grey")
        plt.tight_layout()
        fig.savefig('{p}{n}negentropy-itTV.png'.format(p=data_info.Meta.Path, 
                                                       n=data_info.Meta.Name)) 

        fig, ax1 = plt.subplots(figsize=(10,4))

        color = 'tomato'
        ax1.set_title('$\lambda$ = {}'.format(args.lambda_tv))
        ax1.set_xlabel(r'Iteration')
        ax1.set_ylabel(
            ' Iteration < $\phi_\mathrm{c}$ - $\phi_\mathrm{TV}$ > ($^\circ$)'
            )
        ax1.plot(np.arange(N), deltaphis[1:], color=color, linewidth=5)
        ax1.tick_params(axis='y')
        fig.set_tight_layout(True)
        fig.savefig('{p}{n}non-cum-phi-change.png'.format(p=data_info.Meta.Path,
                                                          n=data_info.Meta.Name))
        
        print('DONE')
        
        """
        fig, ax = plt.subplots(figsize=(10,5))
        ax.set_title('$\lambda$ = {}'.format(L))
        ax.set_xlabel(r'Iteration')
        ax.set_ylabel(
            r'Cumulative < $\phi_\mathrm{c}$ - $\phi_\mathrm{TV}$ > ($^\circ$)'
            )
        ax.plot(
            np.arange(N), np.mean(deltaphi, axis=1), 
            color='orangered', linewidth=5
            )
        plt.tight_layout()
        fig.savefig('{p}{n}cum-phi-change.png'.format(p=path, n=name))

        fig, ax = plt.subplots(figsize=(6,5))
        ax.set_title('$\lambda$ = {}'.format(L))
        ax.set_xlabel(r'1/dHKL (${\AA}^{-1}$)')
        ax.set_ylabel(r'TV Projection Magnitude (TV$_\mathrm{proj}$)')
        ax.scatter(1/og_mtz.compute_dHKL()["dHKL"][flags], deltaproj[N-1], 
                   color='black', 
                   alpha=0.5)
        
        res_mean, data_mean = dsutils.resolution_shells(
            deltaproj[N-1], 1/itmtz[0].compute_dHKL()["dHKL"][flags], 15
            )
        ax.plot(res_mean, data_mean, linewidth=3, linestyle='--', color='orangered')
        plt.tight_layout()
        fig.savefig('{p}{n}tv-err-dhkl.png'.format(p=path, n=name))

        fig, ax = plt.subplots(figsize=(6,5))
        ax.set_title('$\lambda$ = {}'.format(L))
        ax.set_xlabel(r'1/dHKL (${\AA}^{-1}$)')
        ax.set_ylabel(
            r'Cumulative < $\phi_\mathrm{c}$ - $\phi_\mathrm{TV}$ > ($^\circ$)'
            )
        ax.scatter(
            1/og_mtz.compute_dHKL()["dHKL"], np.abs(cum_phase_changes[N-1]), 
            color='orangered', alpha=0.05)
        
        res_mean, data_mean = dsutils.resolution_shells(
            np.abs(cum_phase_changes[N-1]), 1/og_mtz.compute_dHKL()["dHKL"], 
            15)
        ax.plot(res_mean, data_mean, linewidth=3, linestyle='--', color='black')
        plt.tight_layout()
        fig.savefig('{p}{n}cum-phase-change-dhkl.png'.format(p=path, n=name))

        fig, ax = plt.subplots(figsize=(6,5))
        ax.set_title('$\lambda$ = {}'.format(L))
        ax.set_xlabel(r'|Fobs|')
        ax.set_ylabel(
            r'Cumulative < $\phi_\mathrm{c}$ - $\phi_\mathrm{TV}$ > ($^\circ$)'
            )
        ax.scatter(
            og_mtz["WDF"], np.abs(cum_phase_changes[N-1]), color='mediumpurple',
            alpha=0.5)
        plt.tight_layout()
        fig.savefig('{p}{n}cum-phase-change-Fobs.png'.format(p=path, n=name))
    """

if __name__ == "__main__":
    main()