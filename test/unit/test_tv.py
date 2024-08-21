import reciprocalspaceship as rs
from meteor import tv


def test_tv_denoise_difference_map_smoke(flat_difference_map: rs.DataSet) -> None:
    # test sequence pf specified lambda
    tv.tv_denoise_difference_map(
        difference_map_coefficients=flat_difference_map,
        lambda_values_to_scan=[1.0, 2.0],
    )

    # test golden optimizer
    tv.TV_LAMBDA_RANGE = (1.0, 1.01)
    tv.tv_denoise_difference_map(
        difference_map_coefficients=flat_difference_map,
    )


def test_tv_denoise_difference_map_golden(): ...


def test_tv_denoise_difference_map_specific_lambdas(): ...
