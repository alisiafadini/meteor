import argparse
from   tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from meteor import io
from meteor import dsutils
from meteor import maps
from meteor import tv

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
        "-fl",
        "--flags",
        type=str,
        default=None,
        help="If set, label for Rfree flags to use as test set"
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

    # Map writing parameters
    map_res     = 6

    # Parse commandline arguments
    args = parse_arguments()
    
    path              = os.path.split(args.mtz[0])[0]
    name              = os.path.split(args.mtz[0])[1].split('.')[0]
    cell, space_group = io.get_pdbinfo(args.refpdb[0])
    
    print('%%%%%%%%%% ANALYZING DATASET : \
          {n} in {p} %%%%%%%%%%%'.format(n=name, p=path))
    print('CELL         : {}'.format(cell))
    print('SPACEGROUP   : {}'.format(space_group))
    
    # Apply resolution cut if specified
    if args.highres is not None:
        high_res = args.highres
    else:
        high_res = np.min(io.load_mtz(args.mtz[0]).compute_dHKL()["dHKL"])
    
    # Read in mtz file
    og_mtz = io.load_mtz(args.mtz[0])
    og_mtz = og_mtz.loc[og_mtz.compute_dHKL()["dHKL"] > high_res]
    og_mtz = og_mtz.loc[og_mtz.compute_dHKL()["dHKL"] < 25]

    # Use own R-free flags set if specified
    if args.flags is not None:
        og_mtz = og_mtz.loc[:, [args.mtz[1], args.mtz[2], args.mtz[3], args.flags]]  
        flags  = og_mtz[args.flags] == 0  

    else:
        og_mtz = og_mtz.loc[:, [args.mtz[1], args.mtz[2], 
                                args.mtz[3], args.mtz[4], args.mtz[5]]]  
        flags  = np.random.binomial(1, 0.03, 
                                    og_mtz[args.mtz[1]].shape[0]).astype(bool)      

    # scale second dataset ('on') and the first ('off') \
    # to FCalcs, and calculate deltaFs (weighted or not)
    og_mtz, ws = maps.find_w_diffs(og_mtz, args.mtz[2], 
                                   args.mtz[1], args.mtz[5], 
                                   args.mtz[4], args.refpdb[0], 
                                   high_res, path, args.alpha)

    #in case of calculated structure factors:
    #og_mtz["light-phis"]  = mtr.load_mtz(args.mtz[0])["light-phis"] 

    proj_mags         = []
    entropies         = []
    phase_changes     = []
    cum_phase_changes = []
    
    #in case of calculated structure factors:
    #ph_err_corrs      = []

    N = 100
    L = args.lambda_tv  # noqa: E741

    print("lambda is ", L)

    with tqdm(total=N) as pbar:
        for i in np.arange(N) + 1 :
            if i == 1:
                new_amps, new_phases, proj_mag, entropy, phi_change, z =tv.TV_iteration(
                        og_mtz,
                        "WDF",
                        args.mtz[3],
                        "scaled_on",
                        "scaled_off",
                        args.mtz[3],
                        map_res,
                        cell,
                        space_group,
                        flags,
                        L,
                        high_res,
                        ws
                    )

                cum_phase_change = np.abs(
                    np.array(
                        dsutils.positive_Fs(
                            og_mtz,
                            args.mtz[3],
                            "WDF",
                            "phases-pos",
                            "diffs-pos"
                        )["phases-pos"] - new_phases
                    )
                )

                #ph_err_corr          = np.abs(new_phases - \
                # np.array(mtr.positive_Fs(og_mtz, "light-phis",
                # \ "diffs", "phases-pos", "diffs-pos")["phases-pos"])) 

                cum_phase_change     = dsutils.adjust_phi_interval(cum_phase_change)
                #ph_err_corr          = mtr.adjust_phi_interval(ph_err_corr)

                og_mtz["new_amps"]   = new_amps
                og_mtz["new_amps"]   = og_mtz["new_amps"].astype("SFAmplitude")
                og_mtz["new_phases"] = new_phases
                og_mtz["new_phases"] = og_mtz["new_phases"].astype("Phase")
                og_mtz.write_mtz("{name}_TVit{i}_{l}.mtz".format(name=name, i=i, l=L))

            else :
                new_amps, new_phases, proj_mag, entropy, phi_change, z =tv.TV_iteration(
                    og_mtz,
                    "new_amps",
                    "new_phases",
                    "scaled_on",
                    "scaled_off",
                    args.mtz[3],
                    map_res,
                    cell,
                    space_group,
                    flags,
                    L,
                    high_res,
                    ws
                )
                    
                cum_phase_change = (
                    np.abs(
                        np.array(
                            dsutils.positive_Fs(
                                og_mtz, 
                                args.mtz[3], 
                                "WDF", 
                                "phases-pos", 
                                "diffs-pos"
                            )["phases-pos"] - new_phases
        )
    )
)                
                
                cum_phase_change     = dsutils.adjust_phi_interval(cum_phase_change)

                og_mtz["new_amps"]   = new_amps
                og_mtz["new_amps"]   = og_mtz["new_amps"].astype("SFAmplitude")
                og_mtz["new_phases"] = new_phases
                og_mtz["new_phases"] = og_mtz["new_phases"].astype("Phase")

                og_mtz["new-light-phi"] = z
                og_mtz["new-light-phi"] = og_mtz["new-light-phi"].astype("Phase")

                og_mtz.write_mtz("{name}_TVit{i}_{l}.mtz".format(name=name, i=i, l=L))
            
            # Track projection magnitude and phase change for each iteration
            proj_mags.append(proj_mag)
            entropies.append(entropy)
            phase_changes.append(phi_change)
            cum_phase_changes.append(cum_phase_change)
            #ph_err_corrs.append(ph_err_corr)
            pbar.update()

            #np.save("{name}_TVit{i}_{l}-Z-mags.npy".format(name=name, i=i, l=l), z)
    print("FINAL ENTROPY VALUE ", entropies[-1])
    np.save("{name}_TVit{i}_{l}-entropies.npy".format(name=name, i=i, l=L), entropies)

    print("SAVING FINAL MAP")
    finmap = dsutils.map_from_Fs(og_mtz, "new_amps", "new_phases", 6)
    finmap.write_ccp4_map("{name}_TVit{i}_{l}.ccp4".format(name=name, i=i, l=L))    

    # Optionally plot result for errors and negentropy
    if args.plot is True:

        fig, ax1 = plt.subplots(figsize=(10,4))

        color = 'black'
        ax1.set_title('$\lambda$ = {}'.format(L))
        ax1.set_xlabel(r'Iteration')
        ax1.set_ylabel(r'TV Projection Magnitude (TV$_\mathrm{proj}$)', color=color)
        ax1.plot(np.arange(N-1), np.array(np.mean(np.array(proj_mags), axis=1)/
                                          np.max(np.mean(np.array(proj_mags), 
                                                         axis=1)))[1:], color=color, 
                                                         linewidth=5)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx() 

        color = 'silver'
        ax2.set_ylabel('Negentropy', color='grey')  
        ax2.plot(np.arange(N-1), entropies[1:], color=color, linewidth=5)
        ax2.tick_params(axis='y', labelcolor="grey")
        #ax2.set_xlim(0,0.11)
        plt.tight_layout()
        fig.savefig('{p}{n}non-cum-error-TV.png'.format(p=path, n=name)) 

        fig, ax1 = plt.subplots(figsize=(10,4))

        color = 'tomato'
        ax1.set_title('$\lambda$ = {}'.format(L))
        ax1.set_xlabel(r'Iteration')
        ax1.set_ylabel(
            ' Iteration < $\phi_\mathrm{c}$ - $\phi_\mathrm{TV}$ > ($^\circ$)'
            )
        ax1.plot(np.arange(N), phase_changes, color=color, linewidth=5)
        ax1.tick_params(axis='y')
        fig.set_tight_layout(True)
        fig.savefig('{p}{n}non-cum-phi-change.png'.format(p=path, n=name))

        fig, ax = plt.subplots(figsize=(10,5))
        ax.set_title('$\lambda$ = {}'.format(L))
        ax.set_xlabel(r'Iteration')
        ax.set_ylabel(
            r'Cumulative < $\phi_\mathrm{c}$ - $\phi_\mathrm{TV}$ > ($^\circ$)'
            )
        ax.plot(
            np.arange(N), np.mean(cum_phase_changes, axis=1), 
            color='orangered', linewidth=5
            )
        plt.tight_layout()
        fig.savefig('{p}{n}cum-phi-change.png'.format(p=path, n=name))

        fig, ax = plt.subplots(figsize=(6,5))
        ax.set_title('$\lambda$ = {}'.format(L))
        ax.set_xlabel(r'1/dHKL (${\AA}^{-1}$)')
        ax.set_ylabel(r'TV Projection Magnitude (TV$_\mathrm{proj}$)')
        ax.scatter(1/og_mtz.compute_dHKL()["dHKL"][flags], proj_mags[N-1], 
                   color='black', 
                   alpha=0.5)
        
        res_mean, data_mean = dsutils.resolution_shells(
            proj_mags[N-1], 1/og_mtz.compute_dHKL()["dHKL"][flags], 15
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

    print('DONE')

if __name__ == "__main__":
    main()