import reciprocalspaceship as rs
from meteor.rsmap import Map, _assert_is_map
from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    max_negentropy_kweighted_difference_map,
)
from meteor.tv import tv_denoise_difference_map
from meteor.iterative import iterative_tv_phase_retrieval
from meteor.scale import scale_maps
import argparse
import numpy as np


def parse_args() -> argparse.Namespace:
    ...
    # TODO add type of lambda screen input
    # return parser.parse_args()


def main() -> None:
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

    # First, find improved derivative phases
    new_derivative_map, it_tv_metadata = iterative_tv_phase_retrieval(
        derivative_map_scaled,
        native_map_scaled,
        tv_weights_to_scan=[0.01, 0.05, 0.1],
    )
    print("dev map scaled", derivative_map_scaled)
    print("new dev map", new_derivative_map)
    print(it_tv_metadata)

    # compute diffmap
    diff_map = compute_difference_map_wrapper(
        new_derivative_map,
        native_map_scaled,
        use_k_weighting=args.k_weight_with_fixed_parameter
        or args.k_weight_with_parameter_optimization,
        k_param=args.k_weight_with_fixed_parameter,
        optimize_k=args.k_weight_with_parameter_optimization,
    )

    denoised_map, tv_result = tv_denoise_difference_map(
        diff_map, full_output=True, lambda_values_to_scan=np.linspace(0, 0.1, 50)
    )

    print("Writing output file.. ")  # noqa: T201
    final_map = denoised_map
    final_map.write_mtz(args.output)


if __name__ == "__main__":
    main()
