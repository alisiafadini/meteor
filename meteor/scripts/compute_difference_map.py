import argparse

import reciprocalspaceship as rs

from meteor.diffmaps import (
    compute_difference_map,
    compute_kweighted_difference_map,
    max_negentropy_kweighted_difference_map,
)
from meteor.scale import scale_datasets


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute a difference map using native, derivative, and calculated MTZ files."
    )

    parser.add_argument(
        "--native_mtz",
        nargs=3,
        metavar=("filename", "amplitude_label", "uncertainty_label"),
        required=True,
        help=("Native MTZ file and associated amplitude and uncertainty labels."),
    )

    parser.add_argument(
        "--derivative_mtz",
        nargs=4,
        metavar=("filename", "amplitude_label", "uncertainty_label", "[phase_label]"),
        required=True,
        help=(
            "Derivative MTZ file and associated amplitude, uncertainty labels, and optional phase label."
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


def validate_args(args):
    """Ensure that the k_weight_with_fixed_parameter is valid if provided."""
    if args.k_weight_with_fixed_parameter is not None and not (
        0.0 <= args.k_weight_with_fixed_parameter <= 1.0
    ):
        msg = "The --k_weight_with_fixed_parameter must be between 0 and 1.0."
        raise ValueError(msg)
    return args


def ensure_column_consistency(dataset, original_label, target_label):
    """
    Ensure that the target_label exists in the dataset.
    If original_label and target_label are different, copy the column.
    """
    if original_label != target_label:
        if original_label in dataset:
            dataset[target_label] = dataset[original_label]
        else:
            msg = f"Column {original_label} not found in the dataset."
            raise ValueError(msg)
    return dataset


def compute_difference_map_wrapper(
    dataset,
    native_amp,
    native_unc,
    derivative_amp,
    derivative_phase,
    derivative_unc,
    calc_native_phase,
    use_k_weighting=False,
    k_param=None,
    optimize_k=False,
):
    """Wrapper for computing either a standard difference map or a k-weighted difference map."""
    map_args = {
        "dataset": dataset,
        "native_amplitudes_column": native_amp,
        "native_phases_column": calc_native_phase,
        "native_uncertainty_column": native_unc,
        "derivative_amplitudes_column": derivative_amp,
        "derivative_phases_column": derivative_phase,
        "derivative_uncertainty_column": derivative_unc,
    }

    if use_k_weighting:
        if optimize_k:
            result, opt_k = max_negentropy_kweighted_difference_map(**map_args)
            print(f"Optimal k-parameter determined: {opt_k}")  # noqa: T201
            return result
        return compute_kweighted_difference_map(k_parameter=k_param, **map_args)
    return compute_difference_map(**map_args)


def main():
    """Main script function."""
    args = parse_args()

    # Load native MTZ
    native_mtz_filename, native_amplitude_label, native_uncertainty_label = (
        args.native_mtz
    )
    native_ds = rs.read_mtz(native_mtz_filename)

    # Load derivative MTZ (handle optional phase column)
    derivative_mtz_args = args.derivative_mtz
    (
        derivative_mtz_filename,
        derivative_amplitude_label,
        derivative_uncertainty_label,
    ) = derivative_mtz_args[:3]
    derivative_phi_label = (
        derivative_mtz_args[3] if len(derivative_mtz_args) == 4 else None
    )
    derivative_ds = rs.read_mtz(derivative_mtz_filename)

    # Load calculated native MTZ
    calc_native_mtz_filename, calc_amplitude_label, calc_phase_label = (
        args.calc_native_mtz
    )
    calc_native_ds = rs.read_mtz(calc_native_mtz_filename)

    # Ensure amplitude labels are consistent between datasets
    native_ds = ensure_column_consistency(
        native_ds, native_amplitude_label, calc_amplitude_label
    )

    derivative_ds = ensure_column_consistency(
        derivative_ds, derivative_amplitude_label, calc_amplitude_label
    )

    # Scaling the native dataset to the calculated native
    scaled_native_ds = scale_datasets(
        reference_dataset=calc_native_ds,
        dataset_to_scale=native_ds,
        column_to_compare=native_amplitude_label,
        uncertainty_column=(
            native_uncertainty_label if args.use_uncertainties_to_scale else None
        ),
        weight_using_uncertainties=args.use_uncertainties_to_scale,
    )

    # Scaling the derivative dataset to the calculated native
    scaled_derivative_ds = scale_datasets(
        reference_dataset=calc_native_ds,
        dataset_to_scale=derivative_ds,
        column_to_compare=derivative_amplitude_label,
        uncertainty_column=(
            derivative_uncertainty_label if args.use_uncertainties_to_scale else None
        ),
        weight_using_uncertainties=args.use_uncertainties_to_scale,
    )

    # Combine scaled native and scaled derivative datasets
    combined_ds = scaled_native_ds.combine(scaled_derivative_ds)

    diffmap_ds = compute_difference_map_wrapper(
        combined_ds,
        native_amp=native_amplitude_label,
        native_unc=native_uncertainty_label if native_uncertainty_label else None,
        derivative_amp=derivative_amplitude_label,
        derivative_phase=derivative_phi_label,
        derivative_unc=(
            derivative_uncertainty_label if derivative_uncertainty_label else None
        ),
        calc_native_phase=calc_phase_label,
        use_k_weighting=args.k_weight_with_fixed_parameter
        or args.k_weight_with_parameter_optimization,
        k_param=args.k_weight_with_fixed_parameter,
        optimize_k=args.k_weight_with_parameter_optimization,
    )

    print("Writing output file...")  # noqa: T201
    diffmap_ds.write_mtz(args.output)

    print("Process complete.")  # noqa: T201


if __name__ == "__main__":
    main()
