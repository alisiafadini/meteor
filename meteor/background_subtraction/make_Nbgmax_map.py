import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

import reciprocalspaceship as rs
import seaborn as sns
from tqdm import tqdm
sns.set_context("notebook", font_scale=1.4)

from meteor import io
from meteor import dsutils
from meteor import maps
from meteor import validate


"""
Make a background subtracted map with an optimal Nbg value (Nbg_max).
A local region of interest for the background subtraction needs to be specified.
The naming convention chosen for inputs is 'on' and 'off', such
that the generated difference map will be |F_on| - Nbg_max x |F_off|.
Phases come from a reference structure ('off' state).

Write a background subtracted map file (MTZ) and optionally plot/save the plot from the Nbg_max determination (PNG).

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
    
    parser.add_argument("--plot", help="--plot for optional plotting and saving, --no-plot to skip", action=argparse.BooleanOptionalAction)

    
    return parser.parse_args()

def main():

    #Map writing parameters
    map_res     = 4

    # Parse commandline arguments
    args = parse_arguments()
    
    path              = os.path.split(args.onmtz[0])[0]
    name              = os.path.split(args.onmtz[0])[1].split('.')[0].split('/')[-1]
    
    onmtz             = io.subset_to_FSigF(*args.onmtz, {args.onmtz[1]: "F", args.onmtz[2]: "SIGF"})
    offmtz            = io.subset_to_FSigF(*args.offmtz, {args.offmtz[1]: "F", args.offmtz[2]: "SIGF"})
    cell, space_group = io.get_pdbinfo(args.refpdb[0])
    calc              = io.get_Fcalcs(args.refpdb[0], np.min(onmtz.compute_dHKL()["dHKL"]), path)

    # Join all data
    alldata         = onmtz.merge(offmtz, on=["H", "K", "L"], suffixes=("_on", "_off")).dropna()
    common          = alldata.index.intersection(calc.index).sort_values()
    alldata         = alldata.loc[common].compute_dHKL()
    alldata["PHIC"] = calc.loc[common, "PHIC"]
    alldata["FC"]   = calc.loc[common, "FC"]
        
    # Scale both to FCalcs and write differences
    if args.highres is not None:
        alldata     = alldata.loc[alldata["dHKL"] > args.highres]
        spacing     = args.highres / map_res
        h_res       = args.highres
    
    # Screen different background subtraction values for maximum correlation difference
    h_res     = np.min(alldata["dHKL"])  
    spacing   = h_res / map_res
    Nbgs      = np.linspace(0,1, 100)

    CC_diffs  = []
    CC_locs   = []
    CC_globs  = []
    
    calc_map = dsutils.map_from_Fs(alldata,   "FC"   , "PHIC", map_res) #map to use as reference state

    for Nbg in tqdm(Nbgs) :
        
        alldata, _  = maps.find_w_diffs(alldata, "F_on", "F_off", "SIGF_on", "SIGF_off", args.refpdb[0], h_res, path, args.alpha, Nbg)
        Nbg_map  = dsutils.map_from_Fs(alldata, "WDF", "PHIC", map_res)
        
        CC_diff, CC_loc, CC_glob = validate.get_corrdiff(Nbg_map, calc_map, np.array(args.center).astype(float), args.radius, args.refpdb[0], cell, spacing)
        CC_diffs.append(CC_diff)
        CC_locs.append(CC_loc)
        CC_globs.append(CC_glob)
        alldata.drop(columns=["WDF"])
    
    if args.plot is True:

        #Plot and find Nbg that maximizes this difference
        fig, ax = plt.subplots(2,1, figsize=(8,5.5), tight_layout=True)

        ax[0].plot(Nbgs, CC_locs, color='mediumaquamarine', label=r'R$_\mathrm{loc}$', linewidth=3)
        ax[0].plot(Nbgs, CC_globs, color='lightskyblue', label=r'R$_\mathrm{glob}$', linewidth=3)

        ax[1].plot(Nbgs, CC_diffs, color='silver', linestyle= 'dashed', label=r'R$_\mathrm{glob}$ - R$_\mathrm{loc}$', linewidth=3)
        ax[1].vlines(Nbgs[np.argmax(CC_diffs)], 0.22, 0, 'r', linestyle= 'dashed', linewidth=2.5, label='Max={}'.format(np.round(Nbgs[np.argmax(CC_diffs)], decimals=3)))

        ax[0].set_title('{}'.format(name), fontsize=17)
        ax[0].set_xlabel('N$_{\mathrm{bg}}$', fontsize=17)
        ax[0].set_ylabel('CC')
        ax[1].set_xlabel('N$_{\mathrm{bg}}$', fontsize=17)
        ax[1].set_ylabel('CC Difference')
        ax[0].legend(fontsize=17)
        ax[1].legend(fontsize=17)
        fig.savefig("{p}/{n}_plotCCdiff.pdf".format(p=path, n=name))
        print("Saved {p}/{n}_plotCCdiff.pdf".format(p=path, n=name))
    
    
    #Save map with optimal Nbg
    
    alldata["DF-Nbgmax"] = alldata["scaled_on"] - Nbgs[np.argmax(CC_diffs)] * alldata["scaled_off"]
    alldata["DF-Nbgmax"] = alldata["DF-Nbgmax"].astype("SFAmplitude")
    alldata.write_mtz("{p}/{n}_Nbgmax.mtz".format(p=path, n=name))
    print("Wrote {p}/{n}_Nbgmax.mtz with Nbg of {N}".format(p=path, n=name, N=np.round(Nbgs[np.argmax(CC_diffs)], decimals=3)))
    

if __name__ == "__main__":
    main()

