import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from meteor import io
from meteor import dsutils
from meteor import maps
from meteor import validate
from meteor import classes

import seaborn as sns
from tqdm import tqdm
sns.set_context("notebook", font_scale=1.4)


"""
Make a background subtracted map with an optimal Nbg value (Nbg_max).
A local region of interest for the background subtraction needs to be specified.
The naming convention chosen for inputs is 'on' and 'off', such
that the generated difference map will be |F_on| - Nbg_max x |F_off|.
Phases come from a reference structure ('off' state).

Write a background subtracted map file (MTZ). 
Optionally plot/save the plot from the Nbg_max determination (PNG).

"""

def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "-on",
        "--onmtz",
        nargs=3,
        metavar=("mtz", "data_col", "sig_col"),
        required=True,
        help=("MTZ to be used as 'on' data. Specified as (filename, F, SigF)"),
    )
    
    parser.add_argument(
        "-off",
        "--offmtz",
        nargs=3,
        metavar=("mtz", "data_col", "sig_col"),
        required=True,
        help=("MTZ to be used as 'off' data. Specified as (filename, F, SigF)"),
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
    
    parser.add_argument(
        "-c",
        "--center",
        nargs=3,
        metavar=("x", "y", "z"),
        required=True,
        help=(
            "XYZ coordinates in PDB for the region of interest. "
            "Specified as (x, y, z)."
        ),
    )
    
    # Optional arguments
    
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.0,
        help="alpha value for computing difference map weights (default=0.0)",
    )
    
    parser.add_argument(
        "-d_h",
        "--highres",
        type=float,
        default=None,
        help="If set, high res to truncate maps",
    )

    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=5.0,
        help="Local region radius for background subtraction",
    )
    
    parser.add_argument("--plot", help="--plot for optional plotting and saving, \
                        --no-plot to skip", action=argparse.BooleanOptionalAction)

    
    return parser.parse_args()

def main():

    # Parse commandline arguments
    args = parse_arguments()
    
    # Fill out the dataset information needed
    data_info                   = classes.DataInfo()
    data_info.Meta.Path         = os.path.split(args.onmtz[0])[0]
    data_info.Meta.Name         = os.path.split(args.onmtz[0])[1].split('.')[0]
    data_info.Maps.MapRes       = 6 #Map writing resolution
    data_info.Xtal.Cell, data_info.Xtal.SpaceGroup  = io.get_pdbinfo(args.refpdb[0])
    data_info.Meta.Pdb          = args.refpdb[0]

    print('%%%%%%%%%% ANALYZING DATASET : {n} in {p} %%%%%%%%%%'.format(
                                                    n=data_info.Meta.Name, 
                                                    p=data_info.Meta.Path))
    print('CELL         : {}'.format(data_info.Xtal.Cell))
    print('SPACEGROUP   : {}'.format(data_info.Xtal.SpaceGroup))
    
    onmtz           = io.subset_to_FSigF(*args.onmtz, 
                                        {args.onmtz[1]: "F", args.onmtz[2]: "SIGF"})
    offmtz          = io.subset_to_FSigF(*args.offmtz, 
                                        {args.offmtz[1]: "F", args.offmtz[2]: "SIGF"})

    calc            = io.get_Fcalcs(data_info.Meta.Pdb, 
                                    np.min(onmtz.compute_dHKL()["dHKL"]), 
                                    )

    # Join all data
    alldata         = onmtz.merge(offmtz, 
                                  on=["H", "K", "L"], 
                                  suffixes=("_on", "_off")).dropna()
    common          = alldata.index.intersection(calc.index).sort_values()
    alldata["PHIC"] = calc.loc[common, "PHIC"]
    alldata["FC"]   = calc.loc[common, "FC"]
    alldata         = alldata.loc[common].compute_dHKL()
        
    # Apply resolution cut if specified
    if args.highres is not None:
        data_info.Maps.HighRes = args.highres
    else:
        data_info.Maps.HighRes = np.min(alldata["dHKL"])
    
    alldata   = dsutils.res_cutoff(alldata, 
                                   data_info.Maps.HighRes, 
                                   np.max(alldata["dHKL"]))
    data_info.Maps.MaskSpacing = data_info.Maps.HighRes / data_info.Maps.MapRes 
    
    # Screen different background subtraction values for maximum correlation difference
    Nbgs      = np.linspace(0,1, 100)

    CC_diffs  = []
    CC_locs   = []
    CC_globs  = []
    
    #compute map to use as reference state
    calc_map = dsutils.map_from_Fs(alldata, 
                                   "FC" , 
                                   "PHIC", 
                                   data_info.Maps.MapRes) 
    
    #Define MtzLabels
    data_labels = classes.MtzLabels()
    label_attrs = ["Fon", "SigFon", "Foff", "SigFoff", "PhiC"]
    for idx, attr in enumerate(label_attrs):
            setattr(data_labels, attr, alldata.columns[idx])
    

    for Nbg in tqdm(Nbgs) :
        
        #call scaling to FCalcs
        scaled_mtz = maps.scale_toFCalcs(alldata, 
                                        data_labels, 
                                        data_info.Meta.Pdb, 
                                        data_info.Maps.HighRes)

        #call weighting scheme
        diffs, ws  = maps.find_w_diffs(scaled_mtz, args.alpha, Nbg)
        Nbg_map    = dsutils.map_from_Fs(diffs, "WDF", "PHIC", data_info.Maps.MapRes)
                
        CC_diff, CC_loc, CC_glob = validate.get_corrdiff(Nbg_map, 
                                                        calc_map, 
                                                        np.array(args.center).astype(float),
                                                        args.radius, 
                                                        data_info)
        CC_diffs.append(CC_diff)
        CC_locs.append(CC_loc)
        CC_globs.append(CC_glob)

    #Save map with optimal Nbg
    Nbg_max              = Nbgs[np.argmax(CC_diffs)]
    alldata["DF-Nbgmax"] = ws * (alldata["scaled_on"] - 
                                 Nbg_max * alldata["scaled_off"])
    alldata["DF-Nbgmax"] = alldata["DF-Nbgmax"].astype("SFAmplitude")
    alldata.write_mtz("{p}{n}_Nbgmax.mtz".format(p=data_info.Meta.Path, 
                                                 n=data_info.Meta.Name))
    print("Wrote {p}{n}_Nbgmax.mtz with Nbg of {N}".format(p=data_info.Meta.Path, 
                                                n=data_info.Meta.Name, 
                                                N=np.round(Nbg_max, 
                                                           decimals=3)))
    print("SAVING FINAL MAP")
    finmap = dsutils.map_from_Fs(alldata, "DF-Nbgmax", "PHIC", data_info.Maps.MapRes)
    finmap.write_ccp4_map("{p}{n}_Nbgmax.ccp4".format(p=data_info.Meta.Path, 
                                                      n=data_info.Meta.Name))
    print("Wrote {p}{n}_Nbgmax.ccp4 with Nbg of {N}".format(p=data_info.Meta.Path, 
                                                n=data_info.Meta.Name, 
                                                N=np.round(Nbg_max, 
                                                           decimals=3)))
    
    
    if args.plot is True:

        #Plot and mark Nbg that maximizes this difference
        fig, ax = plt.subplots(2,1, figsize=(8,5.5), tight_layout=True)

        ax[0].plot(Nbgs, CC_locs, 
                   color='mediumaquamarine', 
                   label=r'R$_\mathrm{loc}$', 
                   linewidth=3)
        ax[0].plot(Nbgs, CC_globs, 
                   color='lightskyblue', 
                   label=r'R$_\mathrm{glob}$', 
                   linewidth=3)

        ax[1].plot(Nbgs, CC_diffs, 
                   color='silver', 
                   linestyle= 'dashed', 
                   label=r'R$_\mathrm{glob}$ - R$_\mathrm{loc}$', 
                   linewidth=3)
        ax[1].vlines(Nbgs[np.argmax(CC_diffs)], 0.22, 0, 'r', 
                     linestyle= 'dashed', 
                     linewidth=2.5, 
                     label='Max={}'.format(np.round(Nbg_max, decimals=3)))

        ax[0].set_title('{}'.format(data_info.Meta.Name), fontsize=17)
        ax[0].set_xlabel('N$_{\mathrm{bg}}$', fontsize=17)
        ax[0].set_ylabel('CC')
        ax[1].set_xlabel('N$_{\mathrm{bg}}$', fontsize=17)
        ax[1].set_ylabel('CC Difference')
        ax[0].legend(fontsize=17)
        ax[1].legend(fontsize=17)
        fig.savefig("{p}/{n}_plotCCdiff.png".format(p=data_info.Meta.Path, 
                                                    n=data_info.Meta.Name))
        print("Saved {p}/{n}_plotCCdiff.png".format(p=data_info.Meta.Path, 
                                                    n=data_info.Meta.Name))
    

if __name__ == "__main__":
    main()

