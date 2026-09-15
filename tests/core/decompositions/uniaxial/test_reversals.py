"""Test functions for reversal extraction of one-dimensional signals.

Tests cover:
    1. Reversals of simple signals (endpoints, intermediate points on slopes)
    2. Plateaus (first sample of a plateau, plateaus at the start, end, and on slopes)
    3. Constant and short signals
    4. Input validation (ValueError for wrong shape, empty, or non-finite signals)
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from fatpy.core.decompositions.uniaxial.reversals import find_reversals


@pytest.mark.parametrize(
    "signal, expected",
    [
        ([0.0, 2.0, 1.0, 3.0, 0.0], [0, 1, 2, 3, 4]),  # every sample is a reversal
        ([0.0, 1.0, 2.0, 3.0], [0, 3]),  # monotonic: endpoints only
        ([0.0, 1.0, 2.0, 1.0, 0.0], [0, 2, 4]),  # intermediate points dropped
        ([0.0, 2.0, 2.0, 2.0, 1.0], [0, 1, 4]),  # peak plateau: first sample
        ([1.0, 1.0, 0.0, 2.0], [0, 2, 3]),  # plateau at the start
        ([0.0, 2.0, 1.0, 1.0], [0, 1, 2]),  # plateau at the end
        ([0.0, 1.0, 1.0, 2.0], [0, 3]),  # plateau on a slope is not a reversal
        ([3.0, 3.0, 3.0], [0]),  # constant signal
        ([3.0], [0]),  # single sample
        ([1.0, 2.0], [0, 1]),  # two samples
    ],
)
def test_find_reversals(signal: list[float], expected: list[int]) -> None:
    result = find_reversals(signal)

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.int64


def test_find_reversals_example_history() -> None:
    """Every point A..J of the ten-point example history is a reversal."""
    signal = [1.0, -3.0, 2.0, -1.0, 4.0, -5.0, 3.0, -2.0, 5.0, 0.0]

    np.testing.assert_array_equal(find_reversals(signal), np.arange(10))


def test_find_reversals_accepts_integer_list() -> None:
    np.testing.assert_array_equal(find_reversals([0, 2, 2, 1, 3, 0]), [0, 1, 3, 4, 5])


@pytest.mark.parametrize(
    "signal, message",
    [
        (np.zeros((3, 2)), "one-dimensional"),
        (np.float64(1.0), "one-dimensional"),
        (np.array([]), "at least one sample"),
        (np.array([0.0, np.nan, 1.0]), "NaN or infinite"),
        (np.array([0.0, np.inf, 1.0]), "NaN or infinite"),
    ],
)
def test_find_reversals_invalid_signal_raises(
    signal: NDArray[np.float64], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        find_reversals(signal)
