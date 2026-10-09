"""Signal processing utilities module.

This module provides various functions and classes for signal processing tasks.

Overview:
    A signal is the simplest load object: a unitless value vs time, without
    any physical meaning (the quantity, component and unit are attached by
    `fatpy.data_parsing.loads`). Two kinds of signals are provided:

    - `ConstantAmplitudeSignal`: periodic signal defined by its amplitude,
      mean, waveform (sine, triangle, square), period (or frequency) and
      phase. A ``CONSTANT`` waveform gives a time-independent (static) value.
    - `VariableAmplitudeSignal`: values given at their own time instants,
      never interpolated, for example a measured record read with
      `VariableAmplitudeSignal.from_csv`. Without a time scale (no ``time``
      or ``time_step``) the values form a load sequence.

    Combining several signals (common period, sampling) is done where the
    channels are combined, in `fatpy.data_parsing.loads`; a signal only
    reports the instants of its extremes (`ConstantAmplitudeSignal.
    peak_instants`) so that the sampling can hit them.

Conventions:
    - Time is in seconds (or load steps), frequency in Hz and phase in
      degrees.
    - Time instants are matched within `TIME_PRECISION` (1e-9 s).
    - CSV files (`read_numeric_csv`) are comma separated, UTF-8, with one
      header line of column names; every value is read as float64.

Future work:
    - PSD signal (random loading defined by a power spectral density), as a
      new class following the `Signal` protocol. It provides its own time
      instants through `Signal.instants`, so channels and load cases accept it
      unchanged.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Time resolution [s]: time instants are matched with it.
TIME_PRECISION = 1e-9
# |sin(theta)| below this is a zero crossing of the square wave: theta = k*pi
# is not exactly representable, so sin(2*pi) is about -2.4e-16, not 0.
_SINE_ZERO = 1e-12
# Number of lines given at once to numpy.loadtxt.
_CSV_CHUNK_LINES = 200_000


class SignalError(ValueError):
    """Raised when a signal is inconsistently defined or evaluated."""


class CsvFormatError(ValueError):
    """Raised when a CSV file does not have the expected format.

    The message names the file, and the line and column when they are known.
    """


class Waveform(Enum):
    """Shape of one period of a constant-amplitude signal."""

    SINE = "sine"
    TRIANGLE = "triangle"
    SQUARE = "square"
    CONSTANT = "constant"


def waveform_shape(waveform: Waveform, theta: ArrayLike) -> NDArray[np.float64]:
    r"""Normalized waveform value, in [-1, 1], at the phase angle `theta`.

    ??? abstract "Math Equations"
        $$
        f(\theta) =
        \begin{cases}
        \sin\theta & \text{SINE} \\
        \frac{2}{\pi}\arcsin(\sin\theta) & \text{TRIANGLE} \\
        +1 \text{ if } \sin\theta \ge 0 \text{ else } -1 & \text{SQUARE} \\
        0 & \text{CONSTANT}
        \end{cases}
        $$

    For ``SQUARE``, ``|sin θ| < 1e-12`` counts as zero (value +1), so a cycle
    starts and ends on the same value.

    Args:
        waveform: Shape of the signal.
        theta: Phase angle [rad], any shape.

    Returns:
        Array of the same shape as `theta`.

    Example:
        >>> waveform_shape(Waveform.TRIANGLE, [0.0, np.pi / 2]).tolist()
        [0.0, 1.0]
    """
    angle = np.asarray(theta, dtype=np.float64)
    match waveform:
        case Waveform.SINE:
            return np.sin(angle)
        case Waveform.TRIANGLE:
            return (2.0 / np.pi) * np.arcsin(np.clip(np.sin(angle), -1.0, 1.0))
        case Waveform.SQUARE:
            sine = np.sin(angle)
            return np.where((sine >= 0.0) | (np.abs(sine) < _SINE_ZERO), 1.0, -1.0)
        case Waveform.CONSTANT:
            return np.zeros_like(angle)


class Signal(Protocol):
    """Interface shared by all signals."""

    def evaluate(self, time: ArrayLike) -> NDArray[np.float64]:
        """Signal values at the given time instants [s]."""
        ...

    @property
    def effective_period(self) -> float | None:
        """Period [s] of a periodic signal, ``None`` for a non-periodic one."""
        ...

    @property
    def instants(self) -> NDArray[np.float64] | None:
        """Own time instants [s], ``None`` if any instant can be evaluated."""
        ...


def _check_finite(**parameters: float | None) -> None:
    """Raise `SignalError` naming the first parameter that is NaN or inf."""
    for name, value in parameters.items():
        if value is not None and not math.isfinite(value):
            raise SignalError(f"{name} must be finite, got {value!r}")


def _finite_read_only(name: str, data: ArrayLike) -> NDArray[np.float64]:
    """Finite, read-only float64 copy of `data`.

    Raises:
        SignalError: If `data` contains NaN or inf.
    """
    array = np.array(data, dtype=np.float64)
    bad = np.flatnonzero(~np.isfinite(array))
    if bad.size:
        raise SignalError(
            f"{name} must be finite, got {array.flat[bad[0]]!r} at index {bad[0]}"
        )
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ConstantAmplitudeSignal:
    r"""Periodic signal with constant amplitude and mean.

    ??? abstract "Math Equations"
        $$ x(t) = x_m + x_a \, f\!\left(\frac{2\pi t}{T} +
        \frac{\pi \varphi}{180}\right) $$

        with $f$ the normalized waveform (see `waveform_shape`).

    Attributes:
        amplitude: Amplitude $x_a \ge 0$.
        mean: Mean value $x_m$.
        waveform: Shape of one period.
        period: Period $T$ [s]. Exactly one of `period` and `frequency` is
            given, none for a ``CONSTANT`` waveform.
        frequency: Frequency $1/T$ [Hz].
        phase: Phase shift $\varphi$ [deg].

    Example:
        >>> sig = ConstantAmplitudeSignal(200.0, 50.0, frequency=1.0)
        >>> sig.evaluate([0.0, 0.25, 0.75]).tolist()
        [50.0, 250.0, -150.0]
    """

    amplitude: float = 0.0
    mean: float = 0.0
    waveform: Waveform = Waveform.SINE
    period: float | None = None
    frequency: float | None = None
    phase: float = 0.0

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            SignalError: If a parameter is not finite, if the amplitude is
                negative, if a ``CONSTANT`` waveform has a period, a frequency
                or a non-zero amplitude, or if another waveform does not have
                exactly one positive period or frequency.
        """
        _check_finite(
            amplitude=self.amplitude,
            mean=self.mean,
            phase=self.phase,
            period=self.period,
            frequency=self.frequency,
        )
        if self.amplitude < 0.0:
            raise SignalError(f"amplitude must be >= 0, got {self.amplitude!r}")
        if self.waveform is Waveform.CONSTANT:
            if self.period is not None or self.frequency is not None:
                raise SignalError(
                    "A CONSTANT waveform has no period or frequency, got "
                    f"period={self.period!r}, frequency={self.frequency!r}"
                )
            if self.amplitude != 0.0:
                raise SignalError(
                    "A CONSTANT waveform requires amplitude = 0, "
                    f"got {self.amplitude!r}"
                )
            return
        if (self.period is None) == (self.frequency is None):
            raise SignalError(
                "Exactly one of period or frequency must be given, got "
                f"period={self.period!r}, frequency={self.frequency!r}"
            )
        for name, value in (("period", self.period), ("frequency", self.frequency)):
            if value is not None and value <= 0.0:
                raise SignalError(f"{name} must be positive, got {value!r}")

    @property
    def effective_period(self) -> float | None:
        """Period [s], ``None`` for a ``CONSTANT`` waveform."""
        if self.period is not None:
            return self.period
        if self.frequency is not None:
            return 1.0 / self.frequency
        return None

    @property
    def instants(self) -> None:
        """Always ``None``: the signal can be evaluated at any instant."""
        return None

    def evaluate(self, time: ArrayLike) -> NDArray[np.float64]:
        """Signal values at the given time instants.

        Args:
            time: Time instants [s], any shape.

        Returns:
            Array of the same shape as `time`.
        """
        t = np.asarray(time, dtype=np.float64)
        period = self.effective_period
        if period is None:
            return np.full_like(t, self.mean)
        theta = 2.0 * np.pi * t / period + np.deg2rad(self.phase)
        return self.mean + self.amplitude * waveform_shape(self.waveform, theta)

    def peak_instants(self, duration: float) -> NDArray[np.float64]:
        r"""Instants of the maxima and minima in ``[0, duration]``.

        Sine and triangle waves reach their maximum at
        $\theta = \pi/2 + 2k\pi$ and their minimum at $\theta = 3\pi/2 +
        2k\pi$, i.e. at $t = T(1/4 - \varphi/360) + kT$ and
        $t = T(3/4 - \varphi/360) + kT$. A time axis containing these
        instants samples the exact extremes of the signal, whatever the
        phase and the number of samples per period.

        A ``SQUARE`` wave is flat between its jumps and a ``CONSTANT`` one
        never changes: any sample hits their extremes, so none is returned.

        Args:
            duration: End of the time interval [s], >= 0.

        Returns:
            Sorted array of shape (k,), empty for ``SQUARE`` and ``CONSTANT``.

        Example:
            >>> sig = ConstantAmplitudeSignal(1.0, period=1.0, phase=90.0)
            >>> sig.peak_instants(1.0).tolist()
            [0.0, 0.5, 1.0]
        """
        period = self.effective_period
        if period is None or self.waveform not in (Waveform.SINE, Waveform.TRIANGLE):
            return np.empty(0)
        instants = []
        for quarter in (0.25, 0.75):
            first = (period * (quarter - self.phase / 360.0)) % period
            if first > period - TIME_PRECISION:  # rounding of an instant at 0
                first = 0.0
            count = math.floor((duration - first + TIME_PRECISION) / period) + 1
            instants.append(first + period * np.arange(max(count, 0)))
        return np.sort(np.concatenate(instants))


@dataclass(frozen=True, eq=False)
class VariableAmplitudeSignal:
    """Signal defined by its values at its own time instants.

    The signal is not interpolated: it can only be evaluated at its own
    instants (see `instants`). Without `time` and `time_step` the values have
    no time scale and form a load sequence (see `is_sequence`).

    Attributes:
        values: Signal values, shape (n,) with n >= 2, finite, read-only.
        time: Time instants [s], finite, strictly increasing, shape (n,),
            read-only.
        time_step: Constant time step [s], used if `time` is not given.
        instants: Resolved time instants, read-only: `time`, or
            ``arange(n) * time_step``, or the indices ``0, 1, 2, ...`` of a
            load sequence.
    """

    values: NDArray[np.float64]
    time: NDArray[np.float64] | None = None
    time_step: float | None = None
    instants: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Store read-only float64 arrays and resolve the time instants.

        Raises:
            SignalError: If values are not 1D with at least two entries, if
                values or time contain NaN or inf, if both `time` and
                `time_step` are given, if `time` does not match `values` or
                is not strictly increasing, or if `time_step` is not finite
                and positive.
        """
        values = _finite_read_only("values", self.values)
        if values.ndim != 1 or values.size < 2:
            raise SignalError(
                f"values must be 1D with at least two entries, got shape {values.shape}"
            )
        object.__setattr__(self, "values", values)

        if self.time is not None and self.time_step is not None:
            raise SignalError(
                "Give either time or time_step, not both, got "
                f"time_step={self.time_step!r}"
            )
        if self.time is not None:
            instants = _finite_read_only("time", self.time)
            if instants.shape != values.shape:
                raise SignalError(
                    f"time and values must have the same shape, got "
                    f"{instants.shape} and {values.shape}"
                )
            steps = np.diff(instants)
            if np.any(steps <= 0.0):
                i = int(np.argmax(steps <= 0.0))
                raise SignalError(
                    "time must be strictly increasing, got "
                    f"{instants[i]!r} then {instants[i + 1]!r} at index {i + 1}"
                )
            object.__setattr__(self, "time", instants)
        else:
            if self.time_step is not None and (
                not math.isfinite(self.time_step) or self.time_step <= 0.0
            ):
                raise SignalError(
                    f"time_step must be finite and positive, got {self.time_step!r}"
                )
            step = 1.0 if self.time_step is None else self.time_step
            instants = np.arange(values.size, dtype=np.float64) * step
            instants.setflags(write=False)
        object.__setattr__(self, "instants", instants)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        value_column: str,
        time_column: str | None = None,
        time_step: float | None = None,
    ) -> "VariableAmplitudeSignal":
        """Read a measured record from a CSV file.

        The file has one header line with the column names (see
        `read_numeric_csv`); other columns are ignored. Without a time
        column the row order gives the instants, ``arange(n) * time_step``
        (or a load sequence without `time_step`).

        Example file (``value_column="F"``, ``time_column="t"``)::

            t,F
            0.0,0.0
            0.1,5000.0
            0.2,-2000.0

        Args:
            path: CSV file.
            value_column: Name of the column holding the signal values.
            time_column: Name of the column holding the time instants [s].
            time_step: Constant time step [s], only without `time_column`.

        Returns:
            The signal, in the order of the file rows.

        Raises:
            CsvFormatError: If the file cannot be read as numbers, or a
                column is missing (the message names the file, line and
                column).
            SignalError: If both `time_column` and `time_step` are given, or
                the values do not form a valid signal (see the class).
        """
        if time_column is not None and time_step is not None:
            raise SignalError(
                "Give either time_column or time_step, not both, got "
                f"time_step={time_step!r}"
            )
        table = read_numeric_csv(path)
        names = [value_column] if time_column is None else [value_column, time_column]
        for name in names:
            if name not in table.header:
                raise CsvFormatError(
                    f"{path}: no column {name!r}, the columns are {list(table.header)}"
                )
        columns = [table.header.index(n) for n in names]
        table.check_finite(columns)
        values = table.data[:, columns[0]]
        try:
            if time_column is None:
                return cls(values, time_step=time_step)
            return cls(values, time=table.data[:, columns[1]])
        except SignalError as err:
            raise SignalError(f"{path}: {err}") from err

    @property
    def effective_period(self) -> float | None:
        """Always ``None``: the signal is not periodic."""
        return None

    @property
    def is_sequence(self) -> bool:
        """``True`` for a load sequence: neither `time` nor `time_step` given."""
        return self.time is None and self.time_step is None

    @property
    def duration(self) -> float:
        """Last instant minus first instant [s] (number of steps for a sequence)."""
        return float(self.instants[-1] - self.instants[0])

    def evaluate(self, time: ArrayLike) -> NDArray[np.float64]:
        """Signal values at some of its own time instants.

        Args:
            time: Time instants [s], any shape; each must match one of
                `instants` within `TIME_PRECISION`.

        Returns:
            Array of the same shape as `time`.

        Raises:
            SignalError: If a requested instant is not one of `instants`
                (the signal is never interpolated).
        """
        t = np.asarray(time, dtype=np.float64)
        known = self.instants
        right = np.clip(np.searchsorted(known, t), 1, known.size - 1)
        index = np.where(t - known[right - 1] < known[right] - t, right - 1, right)
        off = np.abs(known[index] - t) > TIME_PRECISION
        if np.any(off):
            raise SignalError(
                f"Instant {float(t[off].flat[0])!r} s is not one of the signal "
                "instants (no interpolation)"
            )
        return self.values[index]


# -- CSV reading --------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CsvTable:
    """Numeric content of a CSV file.

    Attributes:
        path: The file read.
        header: Column names of the header line, stripped of spaces.
        data: Values, float64, shape (n_rows, n_columns), read-only.
        lines: Line number in the file of each row (header = line 1), shape
            (n_rows,), read-only.
    """

    path: str
    header: tuple[str, ...]
    data: NDArray[np.float64]
    lines: NDArray[np.int64]

    def check_finite(self, columns: Sequence[int] | None = None) -> None:
        """Raise on the first NaN or inf value, naming its line and column.

        Args:
            columns: Indices of the columns to check, all if ``None``.

        Raises:
            CsvFormatError: If a checked value is NaN or inf.
        """
        selected = list(range(len(self.header))) if columns is None else list(columns)
        bad = np.argwhere(~np.isfinite(self.data[:, selected]))
        if bad.size:
            row, col = int(bad[0, 0]), selected[int(bad[0, 1])]
            raise CsvFormatError(
                f"{self.path}, line {int(self.lines[row])}, column "
                f"{self.header[col]!r}: value is {float(self.data[row, col])!r}, "
                "every value must be finite"
            )


def _locate_bad_value(path: str, header: tuple[str, ...], text: str, line: int) -> str:
    """Describe what makes one CSV line unreadable as numbers."""
    fields = text.rstrip("\r\n").split(",")
    if len(fields) != len(header):
        return (
            f"{path}, line {line}: {len(fields)} values for the {len(header)} "
            f"columns {list(header)}"
        )
    for name, value in zip(header, fields, strict=True):
        try:
            float(value)
        except ValueError:
            return f"{path}, line {line}, column {name!r}: non-numeric value {value!r}"
    return ""


def _parse_chunk(
    path: str, header: tuple[str, ...], texts: list[str], lines: list[int]
) -> NDArray[np.float64]:
    """Parse lines of a CSV file as float64, with a precise error message."""
    try:
        return np.loadtxt(
            texts, delimiter=",", dtype=np.float64, ndmin=2, comments=None
        ).reshape(len(texts), len(header))
    except ValueError as err:
        for text, line in zip(texts, lines, strict=True):
            message = _locate_bad_value(path, header, text, line)
            if message:
                raise CsvFormatError(message) from err
        raise CsvFormatError(f"{path}: {err}") from err  # pragma: no cover


def read_numeric_csv(
    path: str | Path, keep: Callable[[str], bool] | None = None
) -> CsvTable:
    """Read a comma-separated file of numbers with one header line.

    Every value is read as float64 (``nan`` and ``inf`` are accepted here,
    see `CsvTable.check_finite`). Blank lines are skipped. The file is read
    in chunks, so `keep` can drop rows before they are converted, e.g. to
    load a subset of a large file.

    Args:
        path: CSV file, UTF-8 (a byte order mark is accepted).
        keep: Called with the text of each data line; the line is kept if it
            returns ``True``. All lines are kept if ``None``.

    Returns:
        The `CsvTable` (possibly with zero rows).

    Raises:
        CsvFormatError: If the file has no header line or empty column names,
            if a line does not have one value per column, or if a value is
            not a number (the message names the file, line and column).
    """
    name = str(path)
    with open(path, encoding="utf-8-sig") as file:
        first = file.readline()
        header = tuple(c.strip() for c in first.strip().split(","))
        if not first.strip() or not all(header):
            raise CsvFormatError(
                f"{name}, line 1: expected a header of column names, got "
                f"{first.strip()!r}"
            )
        chunks: list[NDArray[np.float64]] = [np.empty((0, len(header)))]
        line_numbers: list[int] = []
        texts: list[str] = []
        chunk_lines: list[int] = []
        for line, text in enumerate(file, start=2):
            if not text.strip() or (keep is not None and not keep(text)):
                continue
            texts.append(text)
            chunk_lines.append(line)
            if len(texts) == _CSV_CHUNK_LINES:
                chunks.append(_parse_chunk(name, header, texts, chunk_lines))
                line_numbers += chunk_lines
                texts, chunk_lines = [], []
        if texts:
            chunks.append(_parse_chunk(name, header, texts, chunk_lines))
            line_numbers += chunk_lines
    data = np.concatenate(chunks)
    lines = np.array(line_numbers, dtype=np.int64)
    data.setflags(write=False)
    lines.setflags(write=False)
    return CsvTable(name, header, data, lines)
