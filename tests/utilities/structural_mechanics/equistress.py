import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

# Replace with actual module name
from fatpy.utilities.structural_mechanics.equistress import von_mises_stress


def test_scalar_input() -> None:
    result = von_mises_stress(100, 50, 25, 10, 5, 0)
    expected = np.sqrt(100**2 + 50**2 + 25**2 - 100 * 50 - 100 * 25 - 50 * 25 + 3 * (10**2 + 5**2 + 0**2))
    assert np.isclose(result, expected)


def test_array_input() -> None:
    s11 = np.array([100, 50])
    s22 = np.array([50, 25])
    s33 = np.array([25, 10])
    s12 = np.array([10, 5])
    s13 = np.array([5, 2])
    s23 = np.array([0, 1])

    result = von_mises_stress(s11, s22, s33, s12, s13, s23)
    expected = np.sqrt(s11**2 + s22**2 + s33**2 - s11 * s22 - s11 * s33 - s22 * s33 + 3 * (s12**2 + s13**2 + s23**2))

    assert_array_almost_equal(result, expected)


def test_list_input() -> None:
    s11 = [100, 50]
    s22 = [50, 25]
    s33 = [25, 10]
    s12 = [10, 5]
    s13 = [5, 2]
    s23 = [0, 1]

    result = von_mises_stress(s11, s22, s33, s12, s13, s23)
    expected = np.sqrt(
        np.array(s11) ** 2
        + np.array(s22) ** 2
        + np.array(s33) ** 2
        - np.array(s11) * np.array(s22)
        - np.array(s11) * np.array(s33)
        - np.array(s22) * np.array(s33)
        + 3 * (np.array(s12) ** 2 + np.array(s13) ** 2 + np.array(s23) ** 2)
    )

    assert_array_almost_equal(result, expected)


def test_inconsistent_shapes() -> None:
    s11 = [1, 2]
    s22 = [1, 2]
    s33 = [1, 2]
    s12 = [1, 2]
    s13 = [1, 2]
    s23 = [1]  # Mismatched shape
    with pytest.raises(AssertionError, match="Components' shape is not consistent."):
        von_mises_stress(s11, s22, s33, s12, s13, s23)
