r"""Reversals (peaks and valleys) of one-dimensional signals.

Reversal extraction is the first step of all cycle counting methods: only the peaks
and valleys of a load-time history are relevant for the counting, the samples in
between are not.

Conventions:
    - Signals are one-dimensional arrays of shape (n,), e.g. a stress component or
      a signed equivalent stress history.
    - Returned indices refer to positions in the original input signal.
    - The first and the last sample of a signal are always treated as reversals.
    - For a plateau (consecutive samples with exactly equal values) forming a
      reversal, the index of the first sample of the plateau is used.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_signal(signal: ArrayLike) -> NDArray[np.float64]:
    """Convert the signal to a float array and validate it.

    Args:
        signal: Signal values of shape (n,).

    Returns:
        Array of shape (n,). The signal as float64.

    Raises:
        ValueError: If the signal is not one-dimensional, is empty, or contains
            non-finite values.
    """
    signal_arr = np.asarray(signal, dtype=np.float64)

    if signal_arr.ndim != 1:
        raise ValueError("Signal must be a one-dimensional array of shape (n,).")
    if signal_arr.size == 0:
        raise ValueError("Signal must contain at least one sample.")
    if not np.all(np.isfinite(signal_arr)):
        raise ValueError("Signal must not contain NaN or infinite values.")

    return signal_arr


def _find_reversals(signal: NDArray[np.float64]) -> NDArray[np.int64]:
    """Find reversal indices of an already validated signal."""
    # Drop repeated samples of plateaus, keeping the first sample of each plateau
    keep = np.concatenate(([True], np.diff(signal) != 0))
    candidates = np.flatnonzero(keep)

    if candidates.size <= 2:
        return candidates.astype(np.int64)

    slope_sign = np.sign(np.diff(signal[candidates]))
    is_turn = slope_sign[:-1] != slope_sign[1:]

    reversals = np.concatenate(
        (candidates[:1], candidates[1:-1][is_turn], candidates[-1:])
    )
    return reversals.astype(np.int64)


def find_reversals(signal: ArrayLike) -> NDArray[np.int64]:
    r"""Find reversals (peaks and valleys) in a one-dimensional signal.

    A reversal is a peak or a valley, i.e. a point at which the first derivative of
    the signal changes sign. The first and the last samples are treated as reversals
    as well, because the derivative is undefined before (after) them.

    ??? abstract "Math Equations"
        For a signal $S = \{S_0, S_1, \dots, S_n\}$ with plateaus reduced to their
        first sample, the reversal sequence $R = \{R_0, \dots, R_m\}$ is extracted
        together with the index mapping $I = \{i_0, \dots, i_m\}$:

        $$ R_k = S_{i_k}, \qquad i_0 = 0, \qquad
        (S_{i_k} - S_{i_{k-1}})(S_{i_{k+1}} - S_{i_k}) < 0 $$

    Args:
        signal: Signal values of shape (n,).

    Returns:
        Array of shape (m,). Indices of reversals in the original signal, in
            ascending order.

    Raises:
        ValueError: If the signal is not one-dimensional, is empty, or contains
            non-finite values.

    Example:
        ```python
        find_reversals([0.0, 2.0, 2.0, 1.0, 3.0, 0.0])
        # array([0, 1, 3, 4, 5])
        ```
    """
    return _find_reversals(_validate_signal(signal))
