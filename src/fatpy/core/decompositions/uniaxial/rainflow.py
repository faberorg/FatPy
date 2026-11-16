r"""Rainflow cycle counting methods.

Rainflow counting decomposes a variable amplitude load-time history into closed
hysteresis loops (full cycles) and unclosed reversals (half cycles). Each method in
this module identifies the cycles by the indices of their limiting samples in the
original signal, so that cycle ranges, means, or any other quantity recorded together
with the signal can be looked up afterwards.

Conventions:
    - Signals are one-dimensional arrays of shape (n,), e.g. a stress component or
      a signed equivalent stress history.
    - Reversals are extracted with
      [find_reversals][fatpy.core.decompositions.uniaxial.reversals.find_reversals]:
      the first and the last samples are reversals, and a plateau is represented by
      its first sample.
    - Every method returns a tuple `(cycle_indices, cycle_counts)`:
        - `cycle_indices`: array of shape (c, 2) with the indices `(start, end)` of
          each counted range in the original signal; the signal values of a cycle are
          therefore obtained as `signal[cycle_indices]`.
        - `cycle_counts`: array of shape (c,) with 1.0 for a full cycle and 0.5 for
          a half cycle.
    - A history is non-periodic by default, i.e. it is applied once. A periodic
      history is one block of a repeatedly applied loading: its end is followed by
      its start.

Implemented methods:
    - `calc_rainflow_cycles_three_point`: three-point counting technique for
      non-periodic and periodic histories.

Planned methods:
    - `calc_rainflow_cycles_four_point`: four-point counting technique with
      residue handling (Amzallag et al., 1994).
    - `calc_rainflow_cycles_hcm`: HCM counting algorithm based on the memory
      behaviour of the material (Clormann and Seeger, 1986).

References:
    AMZALLAG, C., GEREY, J., ROBERT, J., BAHUAUD, J. Standardization of the
    rainflow counting method for fatigue analysis. International Journal of
    Fatigue, 1994, 16(4), 287-293.
    <https://doi.org/10.1016/0142-1123(94)90343-3>

    CLORMANN, U. H., SEEGER, T. RAINFLOW-HCM. Ein Zählverfahren für
    Betriebsfestigkeitsnachweise auf werkstoffmechanischer Grundlage. Stahlbau,
    1986, 55(3), 65-71.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fatpy.core.decompositions.uniaxial.reversals import (
    _find_reversals,
    _validate_signal,
)

FULL_CYCLE = 1.0
HALF_CYCLE = 0.5


def _rotate_to_extreme(signal: NDArray[np.float64]) -> NDArray[np.int64]:
    """Sample order of one periodic block starting and ending at its extreme.

    If the last sample equals the first one, both represent the same point of the
    repeated history and the last sample is dropped. The block is then rotated to
    start at the first occurrence of its maximum or minimum, whichever comes first,
    and closed by repeating that sample at the end.

    Args:
        signal: Validated signal of shape (n,).

    Returns:
        Array of shape (n_block + 1,). Indices into `signal` in the rotated order.
    """
    block_length = signal.size
    if block_length > 1 and signal[-1] == signal[0]:
        block_length -= 1

    block = signal[:block_length]
    start = min(int(np.argmax(block)), int(np.argmin(block)))

    return np.concatenate(
        (np.arange(start, block_length), np.arange(0, start + 1))
    ).astype(np.int64)


def _count_three_point(
    values: list[float],
    positions: list[int],
    starting_point_rule: bool,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Three-point counting of a reversal sequence.

    Args:
        values: Values of the reversals.
        positions: Indices of the reversals in the original signal.
        starting_point_rule: If True, a range containing the starting point is
            counted as a half cycle and only its first point is discarded.

    Returns:
        Tuple (cycle_indices, cycle_counts), see the module conventions.
    """
    starts: list[int] = []
    ends: list[int] = []
    counts: list[float] = []

    # Stack of reversal numbers (into values/positions) not yet discarded.
    # stack[head] is the starting point S; discarding S only advances head.
    stack: list[int] = []
    head = 0

    for k in range(len(values)):
        stack.append(k)

        while len(stack) - head >= 3:
            r1, r2, r3 = stack[-3], stack[-2], stack[-1]
            range_x = abs(values[r3] - values[r2])
            range_y = abs(values[r2] - values[r1])

            if range_x < range_y:
                break

            starts.append(positions[r1])
            ends.append(positions[r2])

            if starting_point_rule and len(stack) - head == 3:
                # Range Y contains the starting point S
                counts.append(HALF_CYCLE)
                head += 1
            else:
                counts.append(FULL_CYCLE)
                del stack[-3:-1]

    remaining = stack[head:]
    for r1, r2 in zip(remaining[:-1], remaining[1:], strict=True):
        starts.append(positions[r1])
        ends.append(positions[r2])
        counts.append(HALF_CYCLE)

    cycle_indices = np.column_stack(
        (np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64))
    )
    cycle_counts = np.array(counts, dtype=np.float64)

    return cycle_indices, cycle_counts


def calc_rainflow_cycles_three_point(
    signal: ArrayLike,
    periodic: bool = False,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    r"""Count cycles using the three-point rainflow counting technique.

    Reversals are extracted first, then processed one by one while the three most
    recent reversals that have not been discarded are checked.

    - **Non-periodic history** (`periodic=False`): the history is applied once.
      A range that contains the starting point of the history cannot close a
      hysteresis loop; it is counted as a half cycle. Ranges remaining at the end
      are counted as half cycles as well.
    - **Periodic history** (`periodic=True`): the signal is one block of a
      repeatedly applied loading. The block is rearranged to start and end at its
      maximum or minimum (whichever occurs first), so that every range closes a
      hysteresis loop and only full cycles are counted. The counts refer to one
      block.

    ??? abstract "Math Equations"
        For the three most recent points $R_{k-2}, R_{k-1}, R_k$ that have not been
        discarded, the ranges are

        $$ Y = |R_{k-1} - R_{k-2}|, \qquad X = |R_k - R_{k-1}| $$

        If $X < Y$, the next reversal is read. If $X \ge Y$, range $Y$ is counted
        and reported by the original signal indices $(i_{k-2}, i_{k-1})$:

        - non-periodic history, $Y$ contains the starting point $S$: half cycle,
          $R_{k-2}$ is discarded and the starting point moves to $R_{k-1}$,
        - otherwise: full cycle, $R_{k-2}$ and $R_{k-1}$ are discarded.

        Each range between consecutive reversals is used exactly once:

        $$ 2 N_{full} + N_{half} = m $$

        where $m$ is the number of ranges between consecutive reversals (of the
        rearranged block for a periodic history, where $N_{half} = 0$).

    Note:
        For a non-periodic history that starts and ends at its absolute maximum or
        minimum, both modes give the same cycles; only the largest cycle is reported
        as two half cycles in the non-periodic mode. A signal starting and ending at
        zero is such a case only if zero is its minimum or maximum (e.g. a
        tension-only loading starting from the unloaded state).

    Args:
        signal: Signal values of shape (n,), e.g. stress in MPa.
        periodic: If True, the signal is treated as one block of a repeatedly
            applied loading. If the last sample equals the first one, both are
            considered the same point of the repeated history. Defaults to False.

    Returns:
        Tuple (cycle_indices, cycle_counts):
            - cycle_indices: Array of shape (c, 2). Each row holds the indices
            (start, end) of the counted range in the original signal, in the
            chronological order of the history. For a non-periodic history
            start < end. For a periodic history start > end marks a cycle that
            continues across the end of the block into the next block. Rows are
            ordered as the cycles are counted; the remaining half cycles come last.
            - cycle_counts: Array of shape (c,). 1.0 for a full cycle and 0.5 for
            a half cycle.

    Raises:
        ValueError: If the signal is not one-dimensional, is empty, or contains
            non-finite values.

    Example:
        ```python
        signal = np.array([0.0, 2.0, 1.0, 3.0, 0.0])
        cycle_indices, cycle_counts = calc_rainflow_cycles_three_point(signal)
        # cycle_indices = [[1, 2], [0, 3], [3, 4]]
        # cycle_counts  = [1.0, 0.5, 0.5]

        values = signal[cycle_indices]  # shape (c, 2)
        ranges = np.abs(values[:, 1] - values[:, 0])
        means = values.mean(axis=1)

        # The same signal as one block of a repeated loading: the last sample is
        # the first sample of the next block, and the range 0-3 closes a loop
        calc_rainflow_cycles_three_point(signal, periodic=True)
        # cycle_indices = [[1, 2], [0, 3]]
        # cycle_counts  = [1.0, 1.0]
        ```
    """
    signal_arr = _validate_signal(signal)

    if periodic:
        order = _rotate_to_extreme(signal_arr)
        rotated = signal_arr[order]
        reversal_positions = order[_find_reversals(rotated)]
    else:
        reversal_positions = _find_reversals(signal_arr)

    values: list[float] = signal_arr[reversal_positions].tolist()
    positions: list[int] = reversal_positions.tolist()

    return _count_three_point(values, positions, starting_point_rule=not periodic)
