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
        "-d_h",
        "--highres",
        type=float,
        default=None,
        help="If set, high res to truncate maps",
    )

    parser.add_argument(
        "--plot",
        help="--plot for optional plotting and saving, --no-plot to skip",
        action=argparse.BooleanOptionalAction,
    )

    return parser.parse_args()


def main():
    # Parse commandline arguments
    map_res = 4

    args = parse_arguments()

    path = os.path.split(args.onmtz[0])[0]
    name = os.path.split(args.onmtz[0])[1].split(".")[0].split("/")[-1]
    cell, space_group = io.get_pdbinfo(args.refpdb[0])

    onmtz = io.subset_to_FSigF(*args.onmtz, {args.onmtz[1]: "F", args.onmtz[2]: "SIGF"})
    offmtz = io.subset_to_FSigF(
        *args.offmtz, {args.offmtz[1]: "F", args.offmtz[2]: "SIGF"}
    )
    # calc = io.get_Fcalcs(args.refpdb[0], np.min(onmtz.compute_dHKL()["dHKL"]), path)
    calc = io.load_mtz(
        "/Users/alisia/Desktop/sept29_files_for_maps/dark_cell83_tjl-r3_SFALL.mtz"
    )

    # Join all data
    alldata = onmtz.merge(offmtz, on=["H", "K", "L"], suffixes=("_on", "_off")).dropna()
    common = alldata.index.intersection(calc.index).sort_values()
    alldata = alldata.loc[common].compute_dHKL()
    # alldata["PHIC"] = calc.loc[common, "PHIC"]
    # alldata["FC"] = calc.loc[common, "FC"]
    alldata["PHIC"] = calc.loc[common, "PHIC_ALL"]
    alldata["FC"] = calc.loc[common, "FC_ALL"]

    alldata.write_mtz("/Users/alisia/Desktop/1nsalldata.mtz")

    # Scale both to FCalcs and write differences
    if args.highres is not None:
        alldata = alldata.loc[alldata["dHKL"] > args.highres]
        h_res = args.highres

    # Screen different background subtraction values for maximum correlation difference
    h_res = np.min(alldata["dHKL"])

    best_alpha_map, errors, entropies = maps.screen_alpha_weight(
        alldata,
        "F_on",
        "F_off",
        "SIGF_on",
        "SIGF_off",
        "PHIC",
        args.refpdb[0],
        path,
        name,
        map_res,
        np.array(args.center).astype("float"),
        hres=h_res,
    )
    best_alpha_map.write_mtz("/Users/alisia/Desktop/test-bestalpha.mtz")

    if args.plot is True:
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = "black"
        ax1.set_xlabel(r"$\lambda$")
        ax1.set_ylabel(
            r"$\sum (\Delta \mathrm{Fobs}_\mathrm{free} - \Delta \mathrm{Fcalc}_\mathrm{free})^2$",
            color=color,
        )
        ax1.plot(np.linspace(0, 1, len(errors)), errors, color=color, linewidth=5)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "silver"
        ax2.set_ylabel("Negentropy", color=color)
        ax2.plot(np.linspace(0, 1, len(entropies)), entropies, color=color, linewidth=5)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()
        # plt.show()
        fig.savefig("{p}{n}alpha-optimize.png".format(p=path, n=name))


if __name__ == "__main__":
    main()
