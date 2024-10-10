
import pytest
import numpy as np

from meteor import testing as mt

def test_phases_allclose():

    close1 = np.array([0.0, 89.9999, 179.9999, 359.9999, 360.9999])
    close2 = np.array([-0.0001, 90.0, 180.0001, 360.0001, 0.9999])
    far = np.array([0.5, 90.5, 180.0, 360.0, 361.0])

    mt.assert_phases_allclose(close1, close2)

    with pytest.raises(AssertionError):
        mt.assert_phases_allclose(close1, far)
