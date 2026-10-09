"""Test functions for the signal module.

Covers the waveforms, the peak instants used for sampling, the constant-
and variable-amplitude signals, the CSV reading of measured records and their
error conditions.
"""

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from fatpy.utils import signal
from fatpy.utils.signal import (
    ConstantAmplitudeSignal,
    CsvFormatError,
    SignalError,
    VariableAmplitudeSignal,
    Waveform,
    read_numeric_csv,
    waveform_shape,
)

RTOL = 1e-9
ATOL = 1e-9  # absolute tolerance for values that should be zero


# -- waveform_shape -----------------------------------------------------------


def test_waveform_shape_values() -> None:
    """Each waveform matches its definition at characteristic angles."""
    theta = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 2])
    np.testing.assert_allclose(
        waveform_shape(Waveform.SINE, theta), np.sin(theta), rtol=RTOL
    )
    np.testing.assert_allclose(
        waveform_shape(Waveform.TRIANGLE, theta),
        [0.0, 0.5, 1.0, -1.0],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_array_equal(
        waveform_shape(Waveform.SQUARE, theta), [1.0, 1.0, 1.0, -1.0]
    )
    np.testing.assert_array_equal(waveform_shape(Waveform.CONSTANT, theta), 0.0)


@pytest.mark.parametrize("waveform", list(Waveform))
def test_waveform_shape_is_bounded(waveform: Waveform) -> None:
    """Every waveform stays within [-1, 1]."""
    values = waveform_shape(waveform, np.linspace(-10.0, 10.0, 1001))
    assert np.all(np.abs(values) <= 1.0)


def test_square_zero_crossings_count_as_positive() -> None:
    """Change 1: |sin| < 1e-12 at k*pi gives +1, so 0 and 2*pi agree."""
    theta = np.array([0.0, np.pi, 2 * np.pi, 4 * np.pi])
    assert np.sin(2 * np.pi) < 0.0  # the float rounding the rule handles
    np.testing.assert_array_equal(waveform_shape(Waveform.SQUARE, theta), 1.0)


@pytest.mark.parametrize("waveform", list(Waveform))
def test_closed_cycle_for_every_waveform(waveform: Waveform) -> None:
    """First and last sample of one period are equal, SQUARE included."""
    if waveform is Waveform.CONSTANT:
        sig = ConstantAmplitudeSignal(mean=50.0, waveform=waveform)
    else:
        sig = ConstantAmplitudeSignal(100.0, 50.0, waveform, period=1.0)
    values = sig.evaluate(np.linspace(0.0, 1.0, 73))
    assert values[-1] == pytest.approx(values[0], rel=RTOL)


# -- ConstantAmplitudeSignal -------------------------------------------------


def test_constant_amplitude_sine_definition() -> None:
    """x(t) = mean + amplitude * sin(2*pi*t/T + phase)."""
    sig = ConstantAmplitudeSignal(amplitude=200.0, mean=50.0, period=2.0, phase=30.0)
    t = np.linspace(0.0, 2.0, 33)
    expected = 50.0 + 200.0 * np.sin(np.pi * t + np.deg2rad(30.0))
    np.testing.assert_allclose(sig.evaluate(t), expected, rtol=RTOL)
    assert sig.instants is None


def test_frequency_and_period_are_equivalent() -> None:
    """A frequency f gives the same signal as the period 1/f."""
    t = np.linspace(0.0, 1.0, 17)
    by_freq = ConstantAmplitudeSignal(amplitude=1.0, frequency=4.0)
    by_period = ConstantAmplitudeSignal(amplitude=1.0, period=0.25)
    assert by_freq.effective_period == pytest.approx(0.25, rel=RTOL)
    np.testing.assert_allclose(by_freq.evaluate(t), by_period.evaluate(t), rtol=RTOL)


def test_constant_waveform_is_static() -> None:
    """A CONSTANT waveform has no period and always returns the mean."""
    sig = ConstantAmplitudeSignal(mean=30.0, waveform=Waveform.CONSTANT)
    assert sig.effective_period is None
    np.testing.assert_array_equal(sig.evaluate([0.0, 1.0, 7.5]), 30.0)


def test_negative_amplitude_raises() -> None:
    """Change 2: a negative amplitude is refused, with its value."""
    with pytest.raises(SignalError, match=r"amplitude must be >= 0, got -1\.0"):
        ConstantAmplitudeSignal(amplitude=-1.0, period=1.0)


@pytest.mark.parametrize(
    "parameter", ["amplitude", "mean", "phase", "period", "frequency"]
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_parameter_raises(parameter: str, bad: float) -> None:
    """Change 2: NaN or inf in any parameter is refused, naming it."""
    params = {"amplitude": 1.0, "period": 1.0} | {parameter: bad}
    if parameter == "frequency":
        del params["period"]
    with pytest.raises(SignalError, match=f"{parameter} must be finite"):
        ConstantAmplitudeSignal(**params)  # type: ignore[arg-type]


# -- VariableAmplitudeSignal -------------------------------------------------


def test_variable_amplitude_default_instants_are_indices() -> None:
    """Without time or time_step the instants are 0, 1, 2, ..."""
    sig = VariableAmplitudeSignal(np.array([3.0, -1.0, 4.0]))
    np.testing.assert_array_equal(sig.instants, [0.0, 1.0, 2.0])
    assert sig.effective_period is None


def test_variable_amplitude_time_step_and_time() -> None:
    """Instants come from time_step or from the explicit time array."""
    values = np.array([0.0, 5.0, -2.0])
    by_step = VariableAmplitudeSignal(values, time_step=0.1)
    by_time = VariableAmplitudeSignal(values, time=np.array([0.0, 0.1, 0.2]))
    np.testing.assert_allclose(by_step.instants, [0.0, 0.1, 0.2], rtol=RTOL)
    np.testing.assert_allclose(by_time.instants, by_step.instants, rtol=RTOL)
    np.testing.assert_array_equal(by_step.evaluate(by_step.instants), values)


def test_variable_amplitude_evaluates_subset_of_instants() -> None:
    """Any of the own instants can be requested, in any order."""
    sig = VariableAmplitudeSignal(np.array([10.0, 20.0, 30.0]), time_step=0.5)
    np.testing.assert_array_equal(sig.evaluate([1.0, 0.0]), [30.0, 10.0])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_variable_amplitude_non_finite_raises(bad: float) -> None:
    """Change 3: NaN or inf in values or time is refused, with its index."""
    with pytest.raises(SignalError, match="values must be finite.*index 1"):
        VariableAmplitudeSignal(np.array([1.0, bad, 3.0]))
    with pytest.raises(SignalError, match="time must be finite.*index 2"):
        VariableAmplitudeSignal(np.ones(3), time=np.array([0.0, 1.0, bad]))


@pytest.mark.parametrize("attribute", ["values", "time", "instants"])
def test_variable_amplitude_arrays_are_read_only(attribute: str) -> None:
    """Change 3: values, time and instants cannot be modified in place."""
    sig = VariableAmplitudeSignal(np.array([1.0, 2.0]), time=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="read-only"):
        getattr(sig, attribute)[0] = 9.0


def test_variable_amplitude_does_not_alias_inputs() -> None:
    """Change 3: the caller's arrays stay writable and are not shared."""
    values = np.array([1.0, 2.0])
    sig = VariableAmplitudeSignal(values)
    values[0] = 9.0
    assert sig.values[0] == 1.0


def test_variable_amplitude_duration() -> None:
    """Change 3: duration is the last instant minus the first."""
    assert VariableAmplitudeSignal(np.zeros(5), time_step=0.1).duration == (
        pytest.approx(0.4, rel=RTOL)
    )
    timed = VariableAmplitudeSignal(np.zeros(3), time=np.array([2.0, 3.0, 7.5]))
    assert timed.duration == pytest.approx(5.5, rel=RTOL)


def test_variable_amplitude_is_sequence_without_time_scale() -> None:
    """Change 3: only values without time or time_step form a load sequence."""
    values = np.array([1.0, 2.0])
    assert VariableAmplitudeSignal(values).is_sequence is True
    assert VariableAmplitudeSignal(values, time_step=0.1).is_sequence is False
    timed = VariableAmplitudeSignal(values, time=np.array([0.0, 1.0]))
    assert timed.is_sequence is False


# -- error conditions ---------------------------------------------------------


def test_constant_waveform_with_period_raises() -> None:
    """A CONSTANT waveform cannot have a period or a frequency."""
    with pytest.raises(SignalError, match="no period"):
        ConstantAmplitudeSignal(waveform=Waveform.CONSTANT, period=1.0)
    with pytest.raises(SignalError, match="no period"):
        ConstantAmplitudeSignal(waveform=Waveform.CONSTANT, frequency=1.0)


def test_constant_waveform_with_amplitude_raises() -> None:
    """A CONSTANT waveform requires amplitude = 0."""
    with pytest.raises(SignalError, match="amplitude = 0, got 1.0"):
        ConstantAmplitudeSignal(amplitude=1.0, waveform=Waveform.CONSTANT)


def test_periodic_without_period_or_frequency_raises() -> None:
    """Neither period nor frequency for a periodic waveform."""
    with pytest.raises(SignalError, match="Exactly one"):
        ConstantAmplitudeSignal(amplitude=1.0)


def test_periodic_with_period_and_frequency_raises() -> None:
    """Both period and frequency."""
    with pytest.raises(SignalError, match="Exactly one"):
        ConstantAmplitudeSignal(amplitude=1.0, period=1.0, frequency=1.0)


def test_non_positive_period_raises() -> None:
    """A zero or negative period or frequency, named with its value."""
    with pytest.raises(SignalError, match=r"period must be positive, got -1\.0"):
        ConstantAmplitudeSignal(amplitude=1.0, period=-1.0)
    with pytest.raises(SignalError, match=r"frequency must be positive, got 0\.0"):
        ConstantAmplitudeSignal(amplitude=1.0, frequency=0.0)


def test_variable_amplitude_single_value_raises() -> None:
    """Fewer than two values."""
    with pytest.raises(SignalError, match="at least two"):
        VariableAmplitudeSignal(np.array([1.0]))


def test_variable_amplitude_time_and_time_step_raises() -> None:
    """Both time and time_step."""
    with pytest.raises(SignalError, match="not both"):
        VariableAmplitudeSignal(
            np.array([1.0, 2.0]), time=np.array([0.0, 1.0]), time_step=1.0
        )


def test_variable_amplitude_time_length_mismatch_raises() -> None:
    """time and values of different lengths."""
    with pytest.raises(SignalError, match="same shape"):
        VariableAmplitudeSignal(np.array([1.0, 2.0]), time=np.array([0.0, 1.0, 2.0]))


def test_variable_amplitude_non_increasing_time_raises() -> None:
    """time not strictly increasing, with the offending values."""
    with pytest.raises(SignalError, match="strictly increasing.*index 2"):
        VariableAmplitudeSignal(np.ones(3), time=np.array([0.0, 1.0, 1.0]))


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
def test_variable_amplitude_invalid_time_step_raises(bad: float) -> None:
    """A zero, negative or non-finite time step."""
    with pytest.raises(SignalError, match="time_step must be finite and positive"):
        VariableAmplitudeSignal(np.array([1.0, 2.0]), time_step=bad)


def test_variable_amplitude_no_interpolation() -> None:
    """Evaluating between the own instants is refused, naming the instant."""
    sig = VariableAmplitudeSignal(np.array([1.0, 2.0]), time_step=1.0)
    with pytest.raises(SignalError, match=r"Instant 0\.5 s.*no interpolation"):
        sig.evaluate([0.5])


def test_signal_error_is_value_error() -> None:
    """SignalError can be caught as a ValueError."""
    assert issubclass(SignalError, ValueError)


# -- immutability / module ----------------------------------------------------


@pytest.mark.parametrize(
    ("obj", "attribute"),
    [
        (ConstantAmplitudeSignal(amplitude=1.0, period=1.0), "amplitude"),
        (VariableAmplitudeSignal(np.array([1.0, 2.0])), "values"),
    ],
)
def test_signals_are_immutable(obj: object, attribute: str) -> None:
    """Signals are frozen dataclasses."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, attribute, 0.0)


def test_module_docstring() -> None:
    """Change 5: typo fixed and future PSD signal documented."""
    doc = signal.__doc__ or ""
    assert "provided, :" not in doc
    assert "Future work:" in doc
    assert "PSD" in doc
    assert "Signal` protocol" in doc
    assert "from_csv" in doc


def test_lcm_and_time_axis_moved_to_loads() -> None:
    """The combination of several signals is not done in the signal module."""
    for name in ("period_lcm", "common_time_axis", "MIN_SAMPLES_PER_PERIOD"):
        assert not hasattr(signal, name)


def test_signal_module_does_not_import_data_parsing() -> None:
    """utils.signal stays independent of fatpy.data_parsing."""
    tree = ast.parse(Path(signal.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [m for m in imported if "data_parsing" in m]


# -- peak instants -------------------------------------------------------------


@pytest.mark.parametrize("waveform", [Waveform.SINE, Waveform.TRIANGLE])
@pytest.mark.parametrize("phase", [0.0, 7.0, 90.0, -33.3, 200.0, 360.0])
def test_peak_instants_are_the_extremes(waveform: Waveform, phase: float) -> None:
    """The signal reaches exactly mean +- amplitude at the peak instants."""
    sig = ConstantAmplitudeSignal(100.0, 20.0, waveform, period=0.4, phase=phase)
    peaks = sig.peak_instants(1.2)
    assert np.all((peaks >= 0.0) & (peaks <= 1.2 + 1e-12))
    values = sig.evaluate(peaks)
    assert np.sum(np.isclose(values, 120.0, rtol=RTOL)) >= 3
    assert np.sum(np.isclose(values, -80.0, rtol=RTOL)) >= 3
    assert np.all(np.isclose(np.abs(values - 20.0), 100.0, rtol=RTOL))


def test_peak_instants_include_both_ends() -> None:
    """A peak at t = 0 is repeated at the end of a whole number of periods."""
    sig = ConstantAmplitudeSignal(1.0, period=1.0, phase=90.0)
    np.testing.assert_allclose(sig.peak_instants(2.0), [0.0, 0.5, 1.0, 1.5, 2.0])


@pytest.mark.parametrize(
    "sig",
    [
        ConstantAmplitudeSignal(1.0, waveform=Waveform.SQUARE, period=1.0),
        ConstantAmplitudeSignal(mean=3.0, waveform=Waveform.CONSTANT),
    ],
)
def test_peak_instants_empty_for_flat_waveforms(
    sig: ConstantAmplitudeSignal,
) -> None:
    """Square and constant waveforms are hit by any sample."""
    assert sig.peak_instants(5.0).size == 0


# -- measured record from CSV --------------------------------------------------


def write(path: Path, text: str) -> Path:
    """Write `text` to `path` and return it."""
    path.write_text(text, encoding="utf-8")
    return path


def test_from_csv_with_time_column(tmp_path: Path) -> None:
    """Values and time instants come from the named columns."""
    text = "t,F,other\n0.0,0,9\n0.1,5000,9\n0.3,-2000,9\n"
    path = write(tmp_path / "rec.csv", text)
    sig = VariableAmplitudeSignal.from_csv(path, "F", time_column="t")
    np.testing.assert_array_equal(sig.values, [0.0, 5000.0, -2000.0])
    np.testing.assert_allclose(sig.instants, [0.0, 0.1, 0.3])


def test_from_csv_without_time_column(tmp_path: Path) -> None:
    """Without time column: row order and time_step, or a load sequence."""
    path = write(tmp_path / "rec.csv", "﻿ F \n1.0\n\n2.0\n3.0\n")
    timed = VariableAmplitudeSignal.from_csv(path, "F", time_step=0.5)
    np.testing.assert_allclose(timed.instants, [0.0, 0.5, 1.0])
    sequence = VariableAmplitudeSignal.from_csv(path, "F")
    assert sequence.is_sequence
    np.testing.assert_array_equal(sequence.values, [1.0, 2.0, 3.0])


def test_from_csv_missing_column_raises(tmp_path: Path) -> None:
    """The error names the file and the available columns."""
    path = write(tmp_path / "rec.csv", "t,F\n0,1\n1,2\n")
    with pytest.raises(CsvFormatError, match=r"rec\.csv: no column 'G'.*\['t', 'F'\]"):
        VariableAmplitudeSignal.from_csv(path, "G")


def test_from_csv_time_column_and_step_raises(tmp_path: Path) -> None:
    """time_column and time_step are exclusive."""
    path = write(tmp_path / "rec.csv", "t,F\n0,1\n1,2\n")
    with pytest.raises(SignalError, match="not both"):
        VariableAmplitudeSignal.from_csv(path, "F", time_column="t", time_step=1.0)


def test_from_csv_nan_raises_with_line_and_column(tmp_path: Path) -> None:
    """A NaN in a used column names the line and the column."""
    path = write(tmp_path / "rec.csv", "t,F\n0,1\n1,nan\n")
    with pytest.raises(CsvFormatError, match=r"rec\.csv, line 3, column 'F'.*nan"):
        VariableAmplitudeSignal.from_csv(path, "F", time_column="t")


def test_from_csv_invalid_signal_names_file(tmp_path: Path) -> None:
    """A non-increasing time column is a SignalError naming the file."""
    path = write(tmp_path / "rec.csv", "t,F\n0,1\n0,2\n")
    with pytest.raises(SignalError, match=r"rec\.csv: time must be strictly"):
        VariableAmplitudeSignal.from_csv(path, "F", time_column="t")


def test_read_numeric_csv_lines_and_filter(tmp_path: Path) -> None:
    """Line numbers count the header and blank lines; keep filters rows."""
    path = write(tmp_path / "t.csv", "a,b\n1,2\n\n3,4\n5,6\n")
    table = read_numeric_csv(path)
    assert table.header == ("a", "b")
    np.testing.assert_array_equal(table.lines, [2, 4, 5])
    assert table.data.dtype == np.float64
    with pytest.raises(ValueError, match="read-only"):
        table.data[0, 0] = 0.0
    kept = read_numeric_csv(path, keep=lambda text: not text.startswith("3"))
    np.testing.assert_array_equal(kept.data, [[1.0, 2.0], [5.0, 6.0]])
    empty = read_numeric_csv(write(tmp_path / "e.csv", "a,b\n"))
    assert empty.data.shape == (0, 2)


def test_read_numeric_csv_in_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chunked reading gives the same table as a single read."""
    rows = "".join(f"{i},{2 * i}\n" for i in range(10))
    path = write(tmp_path / "t.csv", "a,b\n" + rows)
    monkeypatch.setattr(signal, "_CSV_CHUNK_LINES", 3)
    table = read_numeric_csv(path)
    np.testing.assert_array_equal(table.data[:, 1], 2.0 * np.arange(10))
    np.testing.assert_array_equal(table.lines, np.arange(2, 12))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("a,b\n1,2\n3,x\n", r"line 3, column 'b': non-numeric value 'x'"),
        ("a,b\n1,2\n,4\n", r"line 3, column 'a': non-numeric value ''"),
        ("a,b\n1,2\n3,4,5\n", r"line 3: 3 values for the 2 columns"),
        ("a,b\n1\n", r"line 2: 1 values for the 2 columns"),
        ("", r"line 1: expected a header"),
        ("a,,b\n1,2,3\n", r"line 1: expected a header"),
    ],
)
def test_read_numeric_csv_errors(tmp_path: Path, text: str, message: str) -> None:
    """Format errors name the file, the line and the column."""
    path = write(tmp_path / "bad.csv", text)
    with pytest.raises(CsvFormatError, match=r"bad\.csv, " + message):
        read_numeric_csv(path)


def test_csv_format_error_is_value_error() -> None:
    """CsvFormatError can be caught as a ValueError."""
    assert issubclass(CsvFormatError, ValueError)
