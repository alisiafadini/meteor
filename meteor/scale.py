
import reciprocalspaceship as rs


def scale_structure_factors(reference: rs.DataSet, dataset_to_scale: rs.DataSet) -> rs.DataSet:
    """
    Apply an anisotropic scaling so that `dataset_to_scale` is on the same scale as `reference`.

        C * exp{ -(h**2 B11 + k**2 B22 + l**2 B33 +
                    2hk B12 + 2hl  B13 +  2kl B23) }

    This is the same procedure implemented by CCP4's SCALEIT.

    
    .. https://www.ccp4.ac.uk/html/scaleit.html
    """
    ...
