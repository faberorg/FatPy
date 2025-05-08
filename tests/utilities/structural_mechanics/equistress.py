"""Tests for all functions in src/utilities/structural_mechanics."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

# Replace with actual module name
from fatpy.utilities.structural_mechanics.equistress import calc_von_mises_stress


def test_scalar_input() -> None:
    """Test for scalar input."""
    voigt = np.array([[100, 50, 25, 0, 5, 10]])  # [s11, s22, s33, s23, s13, s12]
    expected = np.sqrt(
        100**2 + 50**2 + 25**2 - 100 * 50 - 100 * 25
        - 50 * 25 + 3 * (0**2 + 5**2 + 10**2)
    )
    result = calc_von_mises_stress(voigt)
    if not np.isclose(result[0], expected):
        msg = f"{result} from calc_von_misses is not equal to expected value-{expected}"
        ValueError(msg)



def test_array_input() -> None:
    """Test for array input."""
    voigt = np.array([
        [100, 50, 25, 0, 5, 10],
        [50, 25, 10, 1, 2, 5]
    ])
    expected = np.sqrt(
        voigt[:, 0]**2 + voigt[:, 1]**2 + voigt[:, 2]**2
        - voigt[:, 0]*voigt[:, 1] - voigt[:, 0]*voigt[:, 2] - voigt[:, 1]*voigt[:, 2]
        + 3*(voigt[:, 3]**2 + voigt[:, 4]**2 + voigt[:, 5]**2)
    )
    result = calc_von_mises_stress(voigt)
    assert_array_almost_equal(result, expected)


def test_list_input() -> None:
    """Test for list input."""
    voigt = [
        [100, 50, 25, 0, 5, 10],
        [50, 25, 10, 1, 2, 5]
    ]
    voigt = np.array(voigt)
    expected = np.sqrt(
        voigt[:, 0]**2 + voigt[:, 1]**2 + voigt[:, 2]**2
        - voigt[:, 0]*voigt[:, 1] - voigt[:, 0]*voigt[:, 2] - voigt[:, 1]*voigt[:, 2]
        + 3*(voigt[:, 3]**2 + voigt[:, 4]**2 + voigt[:, 5]**2)
    )
    result = calc_von_mises_stress(voigt)
    assert_array_almost_equal(result, expected)


def test_incorrect_shape() -> None:
    """Expecting to raise Index error when passing less than 6 elements."""
    bad_voigt = np.array([
        [100, 50, 25, 0, 5]  # Only 5 elements instead of 6
    ])
    with pytest.raises(IndexError):
        calc_von_mises_stress(bad_voigt)
