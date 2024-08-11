import argparse
import numpy as np
import os
import reciprocalspaceship as rs


from meteor import meteor_io
from meteor import dsutils
from meteor import tv


"""

Apply a total variation (TV) filter to a map.
The level of filtering (determined by the regularization parameter lambda)
is chosen so as to maximize the map negentropy


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

    # Optional arguments
    parser.add_argument(
        "-d_h",
        "--highres",
        type=float,
        default=None,
        help="If set, high res to truncate maps",
    )

    return parser.parse_args()


def main():

    # Magic numbers
    map_spacing = 4

    # Parse commandline arguments
    args = parse_arguments()

    path = os.path.split(args.mtz[0])[0]
    name = os.path.split(args.mtz[0])[1].split(".")[
        0
    ]  # TODO this seems problematic when full paths are not given

    # Read in mtz file
    og_mtz = meteor_io.subset_to_FandPhi(
        *args.mtz, {args.mtz[1]: "F", args.mtz[2]: "Phi"}
    ).dropna()
    og_mtz = og_mtz.compute_dHKL()

    # Apply resolution cut if specified
    if args.highres is not None:
        high_res = args.highres
    else:
        high_res = np.min(["dHKL"])

        og_mtz = og_mtz.compute_dHKL()
        og_mtz = og_mtz.loc[og_mtz["dHKL"] > high_res]

        # Find and save denoised maps that maximizes the map negentropy
        (
            TVmap_best_err,
            TVmap_best_entr,
            lambda_best_err,
            lambda_best_entr,
            errors,
            entropies,
            amp_change,
            ph_change,
        ) = tv.find_TVmap(og_mtz, "F", "Phi", name, path, map_res, cell, space_group)

        meteor_io.map2mtzfile(
            TVmap_best_err,
            "{n}_TV_{l}_besterror.mtz".format(
                n=name, l=np.round(lambda_best_err, decimals=3)
            ),
            high_res,
        )
        meteor_io.map2mtzfile(
            TVmap_best_entr,
            "{n}_TV_{l}_bestentropy.mtz".format(
                n=name, l=np.round(lambda_best_entr, decimals=3)
            ),
            high_res,
        )

    print(
        "Writing out TV denoised map with weights={lerr} and {lentr}".format(
            lerr=np.round(lambda_best_err, decimals=3),
            lentr=np.round(lambda_best_entr, decimals=3),
        )
    )

    print("DONE.")


if __name__ == "__main__":
    main()
