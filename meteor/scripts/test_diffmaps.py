import reciprocalspaceship as rs
from meteor.rsmap import Map, _assert_is_map
from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    max_negentropy_kweighted_difference_map,
)
from meteor.scale import scale_datasets
import argparse


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
        default="meteor_difference_map.mtz",
        help="Output file name",
    )

    parser.add_argument(
        "--use_uncertainties_to_scale",
        type=bool,
        default=True,
        help="Use uncertainties to scale (default: True)",
    )

    parser.add_argument(
        "--k_weight_with_fixed_parameter",
        type=float,
        default=None,
        help="Use k-weighting with a fixed parameter (float between 0 and 1.0)",
    )

    parser.add_argument(
        "--k_weight_with_parameter_optimization",
        type=bool,
        default=False,
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
    # Parse the command-line arguments
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
        args.derivative_mtz[0],  # Filename
        amplitude_column=args.derivative_mtz[1],  # Amplitude label
        uncertainty_column=args.derivative_mtz[2],  # Uncertainty label
        phase_column=args.derivative_mtz[3],  # Phase label
    )

    # Create the calculated native map from the calculated native MTZ file
    calc_native_map = Map.read_mtz_file(
        args.calc_native_mtz[0],  # Filename
        amplitude_column=args.calc_native_mtz[1],  # Calculated Amplitude label
        phase_column=args.calc_native_mtz[2],  # Calculated Phase label
    )

    # TODO Add a scaling step
    ### insert scaling step code using calc_native_map later

    # calculated diffmap

    diffmap = compute_difference_map_wrapper(
        derivative_map,
        native_map,
        use_k_weighting=args.k_weight_with_fixed_parameter
        or args.k_weight_with_parameter_optimization,
        k_param=args.k_weight_with_fixed_parameter,
        optimize_k=args.k_weight_with_parameter_optimization,
    )

    print("Writing output file...")  # noqa: T201
    diffmap.write_mtz(args.output)


if __name__ == "__main__":
    main()
