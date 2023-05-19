import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from meteor import io
from meteor import validate
from meteor import tv
from meteor import classes
import seaborn as sns
sns.set_context("notebook", font_scale=1.8)




"""
Apply a total variation (TV) filter to a map.
The level of filtering (determined by the regularization parameter lambda)
is chosen so as to minimize the error between denoised and measured amplitudes
or maximize the map negentropy
for a free set ('test' set) of reflections.

Write two denoised map file (MTZ), one for each optimal parameter.
Optionally the plot/save the plot from the lambda determination (PNG).
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
    
    parser.add_argument("--plot", help="--plot for optional plotting and saving, \
                        --no-plot to skip", action=argparse.BooleanOptionalAction)
    
    return parser.parse_args()


def main():

    # Parse commandline arguments
    args = parse_arguments()

    # Fill out the dataset information needed
    data_info = classes.DataInfo()

    data_info.Meta.Path         = os.path.split(args.mtz[0])[0]
    data_info.Meta.Name         = os.path.split(args.mtz[0])[1].split('.')[0]
    data_info.Maps.MapRes       = 4 #Map writing resolution
    data_info.Xtal.Cell, data_info.Xtal.SpaceGroup  = io.get_pdbinfo(args.refpdb[0])
    
    print('%%%%%%%%%% ANALYZING DATASET : {n} in {p} %%%%%%%%%%'.format(
                                                    n=data_info.Meta.Name, 
                                                    p=data_info.Meta.Path))
    print('CELL         : {}'.format(data_info.Xtal.Cell))
    print('SPACEGROUP   : {}'.format(data_info.Xtal.SpaceGroup))
    
    # Apply resolution cut if specified
    if args.highres is not None:
        data_info.Maps.HighRes = args.highres
    else:
        data_info.Maps.HighRes = np.min(io.load_mtz(args.mtz[0]).compute_dHKL()["dHKL"])
    
    data_info.Maps.MaskSpacing = data_info.Maps.HighRes / data_info.Maps.MapRes 

    og_mtz  = io.subset_to_FandPhi(*args.mtz, 
                                    {args.mtz[1]: "F", 
                                    args.mtz[2]: "Phi"}).dropna()

    #if Rfree flags were specified, use these
    if args.flags is not None:
        CV_flags = ~args.flags.astype(bool)     
    #Keep N% of reflections for test set
    else:
        CV_flags = validate.generate_CV_flags((og_mtz.shape[0]))

    #Generate TV maps for a range of lambdas from input mtz
    TV_maps = tv.generate_TVmaps(og_mtz, "F", "Phi", data_info, CV_flags)

    # Find+save denoised maps that (1) minimize CV error (2) maximize map negentropy
    marked_TV_maps = tv.mark_best_TVmap(TV_maps)
    cv_map = next((
        map for map in marked_TV_maps if getattr(
        map.Meta, "BestType", "") == "CV"), None)
    nege_map = next((
        map for map in marked_TV_maps if getattr(
        map.Meta, "BestType", "") == "NEGE"), None)

    io.map_to_mtz(cv_map.Data.MapData, data_info.Maps.HighRes,
                                        '{n}_TV_{l}_besterror.mtz'.format(
                                        n=data_info.Meta.Name,
                                        l=np.round(cv_map.Meta.Lambda, decimals=3)
                                        ),
                                            )
    io.map_to_mtz(nege_map.Data.MapData, data_info.Maps.HighRes,
                                        '{n}_TV_{l}_bestentropy.mtz'.format(
                                        n=data_info.Meta.Name,
                                        l=np.round(nege_map.Meta.Lambda, decimals=3)
                                        ),
                                            )
        
    print('Writing out TV denoised map with weights={lerr} and {lentr}'.format(
                                lerr =np.round(cv_map.Meta.Lambda,   decimals=3), 
                                lentr=np.round(nege_map.Meta.Lambda, decimals=3)))
    

    # Optionally plot result for errors and negentropy
    if args.plot is True:

        errors      = [tv_map.Stats.CVError for tv_map in marked_TV_maps]
        entropies   = [tv_map.Stats.Entropy for tv_map in marked_TV_maps]
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot error
        ax1.set_xlabel(r'$\lambda$')
        ax1.set_ylabel(
        r'$\sum(\Delta\mathrm{Fobs}_\mathrm{free}-\Delta\mathrm{Fcalc}_\mathrm{free})^2$')
        ax1.plot(
            np.linspace(1e-8, 0.1, len(errors)), 
            errors / np.max(errors), 
            color='black', linewidth=5)
        ax1.tick_params(axis='y', labelcolor='black')

        # Plot negentropy
        ax2.set_xlabel(r'$\lambda$')
        ax2.set_ylabel('Negentropy', color='grey')
        ax2.plot(np.linspace(1e-8, 0.1, len(entropies)), 
                 entropies, 
                 color='silver', linewidth=5)
        ax2.tick_params(axis='y', labelcolor='grey')

        fig.tight_layout()
        fig.savefig('{path}{name}lambda-optimize.png'.format(
            path=data_info.Meta.Path, name=data_info.Meta.Name)
            )
        plt.close(fig)
        
        print('DONE.')



if __name__ == "__main__":
    main()


