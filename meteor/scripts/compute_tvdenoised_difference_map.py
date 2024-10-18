import reciprocalspaceship as rs
from meteor.rsmap import Map, _assert_is_map
from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    max_negentropy_kweighted_difference_map,
)
from meteor.tv import tv_denoise_difference_map

from meteor.scale import scale_maps
import argparse
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute a difference map using native, derivative, and calculated MTZ files."
    )

    parser.add_argument(
        "--native_mtz",
        nargs=4,
        metavar=("filename", "amplitude_label", "uncertainty_label", "phase_label"),
        required=True,
        help=(
            "Native MTZ file and associated amplitude, uncertainty labels, and phase label."
        ),
    )

    parser.add_argument(
        "--derivative_mtz",
        nargs=4,
        metavar=("filename", "amplitude_label", "uncertainty_label", "phase_label"),
        required=True,
        help=(
            "Derivative MTZ file and associated amplitude, uncertainty labels, and phase label."
        ),
    )

    parser.add_argument(
        "--calc_native_mtz",
        nargs=3,
        metavar=("filename", "calc_amplitude_label", "calc_phase_label"),
        required=True,
        help=(
            "Calculated native MTZ file and associated calculated amplitude and phase labels."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="meteor_difference_map_TVdenoised.mtz",
        help="Output file name",
    )

    parser.add_argument(
        "--use_uncertainties_to_scale",
        type=bool,
        default=True,
        help="Use uncertainties to scale (default: True)",
    )

    k_weight_group = parser.add_mutually_exclusive_group()

    k_weight_group.add_argument(
        "--k_weight_with_fixed_parameter",
        type=float,
        default=None,
        help="Use k-weighting with a fixed parameter (float between 0 and 1.0)",
    )

    k_weight_group.add_argument(
        "--k_weight_with_parameter_optimization",
        action="store_true",  # This will set the flag to True when specified
        help="Use k-weighting with parameter optimization (default: False)",
    )

    return parser.parse_args()


def compute_difference_map_wrapper(
    derivative_map,
    native_map,
    use_k_weighting=False,
    k_param=None,
    optimize_k=False,
):
    """Wrapper for computing either a standard difference map or a k-weighted difference map."""
    if use_k_weighting:
        if optimize_k:
            result, opt_k = max_negentropy_kweighted_difference_map(
                derivative_map, native_map
            )
            print(f"Optimal k-parameter determined: {opt_k}")  # noqa: T201
            return result
        return compute_kweighted_difference_map(
            derivative_map, native_map, k_parameter=k_param
        )
    return compute_difference_map(derivative_map, native_map)


def main():
    # Parse command-line arguments
    args = parse_args()

    # Create the native map from the native MTZ file
    native_map = Map.read_mtz_file(
        args.native_mtz[0],
        amplitude_column=args.native_mtz[1],
        uncertainty_column=args.native_mtz[2],
        phase_column=args.native_mtz[3],
    )

    # Create the derivative map from the derivative MTZ file
    derivative_map = Map.read_mtz_file(
        args.derivative_mtz[0],
        amplitude_column=args.derivative_mtz[1],
        uncertainty_column=args.derivative_mtz[2],
        phase_column=args.derivative_mtz[3],
    )

    # Create the calculated native map from the calculated native MTZ file
    calc_native_map = Map.read_mtz_file(
        args.calc_native_mtz[0],
        amplitude_column=args.calc_native_mtz[1],
        phase_column=args.calc_native_mtz[2],
    )

    # Scale both to common map calculated from native model
    native_map_scaled = scale_maps(
        reference_map=calc_native_map,
        map_to_scale=native_map,
        weight_using_uncertainties=False,
    )

    derivative_map_scaled = scale_maps(
        reference_map=calc_native_map,
        map_to_scale=derivative_map,
        weight_using_uncertainties=False,  # FC do not have uncertainties
    )

    # Calculate diffmap
    diffmap = compute_difference_map_wrapper(
        derivative_map_scaled,
        native_map_scaled,
        use_k_weighting=args.k_weight_with_fixed_parameter
        or args.k_weight_with_parameter_optimization,
        k_param=args.k_weight_with_fixed_parameter,
        optimize_k=args.k_weight_with_parameter_optimization,
    )

    # Now TV denoise

    denoised_map, tv_result = tv_denoise_difference_map(
        diffmap, full_output=True, lambda_values_to_scan=np.linspace(1e-8, 0.1, 100)
    )
    # denoised_map, tv_result = tv_denoise_difference_map(diffmap, full_output=True)
    print(tv_result)

    print("Writing output file...")  # noqa: T201
    denoised_map.write_mtz(args.output)


if __name__ == "__main__":
    main()
