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
            "PDB to be used as reference ('off') structure. " "Specified as (filename)."
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

    return parser.parse_args()


def main():
    # Map writing parameters
    map_res = 4

    # Parse commandline arguments
    args = parse_arguments()

    path = os.path.split(args.onmtz[0])[0]
    name = os.path.split(args.onmtz[0])[1].split(".")[0].split("/")[-1]

    onmtz = io.subset_to_FSigF(*args.onmtz, {args.onmtz[1]: "F", args.onmtz[2]: "SIGF"})
    offmtz = io.subset_to_FSigF(
        *args.offmtz, {args.offmtz[1]: "F", args.offmtz[2]: "SIGF"}
    )
    cell, space_group = io.get_pdbinfo(args.refpdb[0])
    calc = io.get_Fcalcs(args.refpdb[0], np.min(onmtz.compute_dHKL()["dHKL"]), path)

    # Join all data
    alldata = onmtz.merge(offmtz, on=["H", "K", "L"], suffixes=("_on", "_off")).dropna()
    common = alldata.index.intersection(calc.index).sort_values()
    alldata = alldata.loc[common].compute_dHKL()
    alldata["PHIC"] = calc.loc[common, "PHIC"]
    alldata["FC"] = calc.loc[common, "FC"]

    # Scale both to FCalcs and write differences
    if args.highres is not None:
        alldata = alldata.loc[alldata["dHKL"] > args.highres]
        h_res = args.highres

    else:
        h_res = np.min(alldata["dHKL"])

    alldata, weights = maps.find_w_diffs(
        alldata,
        "F_on",
        "F_off",
        "SIGF_on",
        "SIGF_off",
        args.refpdb[0],
        h_res,
        path,
        args.alpha,
        1.0,
    )

    # Save map with optimal Nbg
    alldata.write_mtz("{p}{n}_diffmap.mtz".format(p=path, n=name))
    print("Wrote {p}{n}_diffmap.mtz".format(p=path, n=name))
    print("SAVING FINAL MAP")
    finmap = dsutils.map_from_Fs(alldata, "WDF", "PHIC", 6)
    finmap.write_ccp4_map("{p}{n}_diffmap.ccp4".format(p=path, n=name))


if __name__ == "__main__":
    main()
