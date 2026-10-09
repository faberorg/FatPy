"""Test functions for the load definition module.

Covers the 7 reference load cases (tension, torsion, in-phase and
out-of-phase combined loading, mean stress, different frequencies, force
history), their strain versions, the sampling of periodic channels (common
period, 72 points per shortest period, exact extremes), the Voigt export, the
superposition of FE results and the error conditions.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from fatpy.data_parsing import loads
from fatpy.data_parsing.loads import (
    DEFAULT_SAMPLES_PER_PERIOD,
    Channel,
    ChannelHistory,
    FEResultFile,
    LoadCase,
    LoadDefinitionError,
    LoadHistory,
    Quantity,
    common_time_axis,
    period_lcm,
    superpose,
    voigt_index,
)
from fatpy.data_parsing.fe_model import (
    FEResultError,
    PlasticStrainError,
    read_fe_csv,
)
from fatpy.utils import voigt
from fatpy.utils.signal import (
    ConstantAmplitudeSignal,
    SignalError,
    VariableAmplitudeSignal,
    Waveform,
)

from fe_csv_data import (  # tests/data_parsing/fe_csv_data.py
    TIMES,
    fe_lines,
    fe_table,
    set_cell,
    write_fe,
    write_lines,
)

RTOL = 1e-9
ATOL = 1e-9  # absolute tolerance for values that should be zero
STRAIN_SCALE = 1e-5  # mm/mm per MPa, to rebuild the stress cases as strains


def build_case(number: int, quantity: Quantity = Quantity.STRESS) -> LoadCase:
    """Reference cases 1 to 6, as stress (MPa) or strain (scaled) channels.

    Channel "axial" is the 11 component (200 amplitude, 1 Hz), channel
    "shear" the 12 component (115 amplitude, 1 Hz).
    """
    scale = STRAIN_SCALE if quantity is Quantity.STRAIN else 1.0
    prefix = quantity.components[0][0]

    def channel(
        name: str,
        component: str,
        amplitude: float,
        frequency: float = 1.0,
        mean: float = 0.0,
        phase: float = 0.0,
    ) -> Channel:
        sig = ConstantAmplitudeSignal(
            amplitude * scale, mean * scale, frequency=frequency, phase=phase
        )
        return Channel(name, quantity, prefix + component, sig)

    axial = channel("axial", "11", 200.0)
    shear = channel("shear", "12", 115.0)
    channels = {
        1: [axial],
        2: [shear],
        3: [axial, shear],
        4: [axial, channel("shear", "12", 115.0, phase=90.0)],
        5: [channel("axial", "11", 200.0, mean=100.0)],
        6: [axial, channel("shear", "12", 115.0, frequency=2.0)],
    }[number]
    return LoadCase(f"case {number}", channels)


FX_VALUES = np.array([0.0, 5000.0, -2000.0, 8000.0, 0.0])


def force_channel(time_step: float | None = 0.1) -> Channel:
    """Case 7: Fx = 0, 5000, -2000, 8000, 0 N (time step 0.1 s by default)."""
    sig = VariableAmplitudeSignal(FX_VALUES, time_step=time_step)
    return Channel("Fx", Quantity.FORCE, "Fx", sig)


def static(name: str = "preload", mean: float = 30.0) -> Channel:
    """Static s22 channel (CONSTANT waveform)."""
    sig = ConstantAmplitudeSignal(mean=mean, waveform=Waveform.CONSTANT)
    return Channel(name, Quantity.STRESS, "s22", sig)


class FailingSignal:
    """Signal that can be defined but never evaluated."""

    @property
    def effective_period(self) -> float | None:
        """Not periodic."""
        return None

    @property
    def instants(self) -> NDArray[np.float64] | None:
        """Own instants 0 and 1 s."""
        return np.array([0.0, 1.0])

    def evaluate(self, time: ArrayLike) -> NDArray[np.float64]:
        """Always fails."""
        raise SignalError("broken signal")


# -- the 7 reference cases ----------------------------------------------------


def test_case1_tension() -> None:
    """s11 sine 200 MPa, 1 Hz: peaks +-200, 73 samples with N = 72."""
    history = build_case(1).history()
    axial = history.get("axial").values
    assert len(history) == 73
    assert history.time[-1] == pytest.approx(1.0, rel=RTOL)
    assert axial.max() == pytest.approx(200.0, rel=RTOL)
    assert axial.min() == pytest.approx(-200.0, rel=RTOL)
    np.testing.assert_allclose(
        axial, 200.0 * np.sin(2 * np.pi * history.time), rtol=RTOL, atol=ATOL
    )
    assert history.get("axial").unit == "MPa"


def test_case2_torsion() -> None:
    """s12 sine 115 MPa: peaks +-115."""
    shear = build_case(2).history().get("shear").values
    assert shear.max() == pytest.approx(115.0, rel=RTOL)
    assert shear.min() == pytest.approx(-115.0, rel=RTOL)


def test_case3_in_phase() -> None:
    """In phase: s12 / s11 is constant (= 115 / 200)."""
    history = build_case(3).history()
    np.testing.assert_allclose(
        history.get("shear").values,
        history.get("axial").values * (115.0 / 200.0),
        rtol=RTOL,
        atol=ATOL,
    )


def test_case4_out_of_phase() -> None:
    """90 deg out of phase: (s11 / 200)^2 + (s12 / 115)^2 = 1."""
    history = build_case(4).history()
    s11 = history.get("axial").values
    s12 = history.get("shear").values
    np.testing.assert_allclose((s11 / 200.0) ** 2 + (s12 / 115.0) ** 2, 1.0, rtol=RTOL)


def test_case5_mean_stress() -> None:
    """Mean 100 MPa, amplitude 200 MPa: max 300 MPa, min -100 MPa."""
    axial = build_case(5).history().get("axial").values
    assert axial.max() == pytest.approx(300.0, rel=RTOL)
    assert axial.min() == pytest.approx(-100.0, rel=RTOL)


def test_case6_different_frequencies() -> None:
    """s11 at 1 Hz and s12 at 2 Hz: cycle of 1 s sampled with 145 points."""
    history = build_case(6).history()
    assert len(history) == 145
    assert history.time[-1] == pytest.approx(1.0, rel=RTOL)
    np.testing.assert_allclose(np.diff(history.time), 0.5 / 72, rtol=RTOL)
    np.testing.assert_allclose(
        history.get("shear").values,
        115.0 * np.sin(4 * np.pi * history.time),
        rtol=RTOL,
        atol=ATOL,
    )


def test_case7_force_history() -> None:
    """Fx with time step 0.1 s: output equals input, at its own instants."""
    history = LoadCase("case 7", [force_channel()]).history()
    np.testing.assert_array_equal(history.get("Fx").values, FX_VALUES)
    np.testing.assert_allclose(history.time, [0.0, 0.1, 0.2, 0.3, 0.4], rtol=RTOL)
    assert history.get("Fx").unit == "N"


@pytest.mark.parametrize("number", range(1, 7))
def test_strain_cases_to_voigt_strain(number: int) -> None:
    """Cases 1-6 rebuilt with strain channels give the scaled stress array."""
    stress = build_case(number).history().to_voigt_stress()
    strain = build_case(number, Quantity.STRAIN).history().to_voigt_strain()
    np.testing.assert_allclose(strain, stress * STRAIN_SCALE, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(strain[:, 1:5], 0.0)


# -- time axis ----------------------------------------------------------------


def test_pragtic_example_time_axis() -> None:
    """Periods 1.5 s and 2 s with N = 4: grid dt = 0.375 s plus the peaks.

    The 17 grid instants are kept; the extremes of the 2 s sine (0.5, 1.5,
    2.5, ... s) are not on the grid and are added.
    """
    channels = [
        Channel("a", Quantity.STRESS, "s11", ConstantAmplitudeSignal(1.0, period=1.5)),
        Channel("b", Quantity.STRESS, "s12", ConstantAmplitudeSignal(1.0, period=2.0)),
    ]
    time = LoadCase("pragtic", channels, samples_per_period=4).time_axis()
    grid = np.arange(17) * 0.375
    peaks_b = 0.5 + np.arange(6) * 1.0
    np.testing.assert_allclose(time, np.unique(np.concatenate([grid, peaks_b])))
    assert time[-1] == pytest.approx(6.0, rel=RTOL)


@pytest.mark.parametrize(
    "waveform", [w for w in Waveform if w is not Waveform.CONSTANT]
)
def test_closed_cycle_for_every_waveform(waveform: Waveform) -> None:
    """Last sample equals first sample for every waveform, SQUARE included."""
    signals = [
        ConstantAmplitudeSignal(100.0, 50.0, waveform, frequency=1.0, phase=30.0),
        ConstantAmplitudeSignal(80.0, -20.0, waveform, frequency=2.0, phase=45.0),
    ]
    channels = [
        Channel("a", Quantity.STRESS, "s11", signals[0]),
        Channel("b", Quantity.STRESS, "s12", signals[1]),
        static(),
    ]
    values = LoadCase("closed", channels).history().values_array()
    np.testing.assert_allclose(values[-1], values[0], rtol=RTOL)


def test_static_only_load_case_has_single_instant() -> None:
    """Only CONSTANT channels: the time axis is [0.0]."""
    history = LoadCase("static", [static()]).history()
    np.testing.assert_array_equal(history.time, [0.0])
    np.testing.assert_array_equal(history.get("preload").values, [30.0])


def test_static_channel_next_to_variable_amplitude() -> None:
    """Change 12: a CONSTANT channel is allowed with a variable-amplitude one."""
    history = LoadCase("preloaded", [force_channel(), static()]).history()
    np.testing.assert_array_equal(history.get("preload").values, 30.0)
    assert len(history) == FX_VALUES.size


# -- change 6: validation -----------------------------------------------------


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_channel_name_raises(name: str) -> None:
    """An empty channel name is refused."""
    with pytest.raises(LoadDefinitionError, match="Channel name must not be empty"):
        Channel(name, Quantity.FORCE, "Fx", force_channel().signal)


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_load_case_name_raises(name: str) -> None:
    """An empty load case name is refused."""
    with pytest.raises(LoadDefinitionError, match="Load case name must not be empty"):
        LoadCase(name, [force_channel()])


def test_load_case_samples_per_period_raises() -> None:
    """samples_per_period below 4 is refused, naming the load case and value."""
    with pytest.raises(LoadDefinitionError, match=r"'coarse'.*>= 4, got 3"):
        LoadCase("coarse", build_case(1).channels, samples_per_period=3)


def test_load_case_fractional_samples_per_period_raises() -> None:
    """A fractional samples_per_period would silently truncate the cycle."""
    with pytest.raises(LoadDefinitionError, match=r"integer >= 4, got 4\.5"):
        LoadCase("coarse", build_case(1).channels, samples_per_period=4.5)  # type: ignore[arg-type]


def test_load_history_length_mismatch_raises() -> None:
    """A LoadHistory needs one value per instant in every channel."""
    channel = ChannelHistory("s", Quantity.STRESS, "s11", "MPa", np.zeros(3))
    with pytest.raises(LoadDefinitionError, match=r"'s' has \(3,\) values"):
        LoadHistory(np.zeros(4), (channel,), "bad", False)


def test_rational_frequencies_share_a_period() -> None:
    """2 Hz and 3 Hz (period 1/3 s) have the common period 1 s."""
    channels = [
        Channel(
            "a", Quantity.STRESS, "s11", ConstantAmplitudeSignal(1.0, frequency=2.0)
        ),
        Channel(
            "b", Quantity.STRESS, "s12", ConstantAmplitudeSignal(1.0, frequency=3.0)
        ),
    ]
    time = LoadCase("2 and 3 Hz", channels).time_axis()
    assert time[-1] == pytest.approx(1.0, rel=RTOL)
    assert time.size == 3 * 72 + 1


# -- change 7: SignalError of a channel ---------------------------------------


def test_history_names_failing_channel() -> None:
    """A SignalError while sampling becomes a LoadDefinitionError."""
    channel = Channel("gauge", Quantity.STRAIN, "e11", FailingSignal())
    with pytest.raises(LoadDefinitionError, match="'gauge': broken signal") as info:
        LoadCase("measured", [channel]).history()
    assert isinstance(info.value.__cause__, SignalError)


# -- change 8: load sequence --------------------------------------------------


def test_is_sequence_only_without_time_scale() -> None:
    """is_sequence comes from VariableAmplitudeSignal.is_sequence."""
    sequence = LoadCase("sequence", [force_channel(time_step=None)])
    timed = LoadCase("timed", [force_channel()])
    assert sequence.is_sequence is True
    assert sequence.history().is_sequence is True
    np.testing.assert_array_equal(sequence.history().time, np.arange(5.0))
    assert timed.is_sequence is False
    assert timed.history().is_sequence is False
    assert build_case(1).history().is_sequence is False


# -- change 9: read-only arrays -----------------------------------------------


def test_history_arrays_are_read_only() -> None:
    """time and channel values of a LoadHistory cannot be modified."""
    history = build_case(4).history()
    with pytest.raises(ValueError, match="read-only"):
        history.time[0] = 1.0
    for channel in history.channels:
        with pytest.raises(ValueError, match="read-only"):
            channel.values[0] = 1.0


# -- changes 10 and 11: Voigt export ------------------------------------------


def test_to_voigt_stress_fills_only_columns_0_and_5() -> None:
    """Case 4: s11 goes to column 0, s12 to column 5, the rest stays zero."""
    history = build_case(4).history()
    stress = history.to_voigt_stress()
    assert stress.shape == (len(history), 6)
    np.testing.assert_array_equal(stress[:, 0], history.get("axial").values)
    np.testing.assert_array_equal(stress[:, 5], history.get("shear").values)
    np.testing.assert_array_equal(stress[:, 1:5], 0.0)
    voigt.check_shape(stress)


def test_to_voigt_stress_rejects_non_stress_channel() -> None:
    """Change 10: a force channel is an error naming the channel."""
    sig = VariableAmplitudeSignal(np.arange(5.0), time_step=0.1)
    channels = [force_channel(), Channel("s", Quantity.STRESS, "s33", sig)]
    history = LoadCase("mixed", channels).history()
    with pytest.raises(LoadDefinitionError, match="'Fx' is force"):
        history.to_voigt_stress()


def test_to_voigt_strain_columns() -> None:
    """Change 11: e11 -> column 0, ..., e12 -> column 5."""
    values = np.arange(6.0) * 1e-4
    channels = [
        Channel(
            f"e{i}",
            Quantity.STRAIN,
            component,
            ConstantAmplitudeSignal(mean=v, waveform=Waveform.CONSTANT),
        )
        for i, (component, v) in enumerate(
            zip(Quantity.STRAIN.components, values, strict=True)
        )
    ]
    strain = LoadCase("all strains", channels).history().to_voigt_strain()
    np.testing.assert_allclose(strain, values[np.newaxis, :], rtol=RTOL)
    voigt.check_shape(strain)


def test_to_voigt_strain_rejects_stress_channel() -> None:
    """Change 11: a stress channel cannot enter a strain array."""
    with pytest.raises(LoadDefinitionError, match="'axial' is stress"):
        build_case(1).history().to_voigt_strain()


@pytest.mark.parametrize("quantity", [Quantity.STRESS, Quantity.STRAIN])
def test_to_voigt_duplicate_component_raises(quantity: Quantity) -> None:
    """Two channels cannot give the same component."""
    first = build_case(1, quantity).channels[0]
    second = Channel("axial 2", quantity, first.component, first.signal)
    history = LoadCase("dup", [first, second]).history()
    export = (
        history.to_voigt_stress
        if quantity is Quantity.STRESS
        else history.to_voigt_strain
    )
    with pytest.raises(
        LoadDefinitionError, match="both channels 'axial' and 'axial 2'"
    ):
        export()


# -- change 12: table ---------------------------------------------------------


def test_to_table_is_a_dict_with_units() -> None:
    """to_table returns a plain dict of columns with units in the keys."""
    history = build_case(4).history()
    table = history.to_table()
    assert type(table) is dict
    assert list(table) == ["time [s]", "axial [MPa]", "shear [MPa]"]
    np.testing.assert_array_equal(table["time [s]"], history.time)
    assert history.values_array().shape == (len(history), 2)


# -- change 13: module docstring ----------------------------------------------


def test_module_docstring_mapping_and_future_work() -> None:
    """The module documents the PragTic mapping and the future work."""
    doc = loads.__doc__ or ""
    assert "PragTic mapping:" in doc
    for line in (
        "load regime -> `LoadCase`",
        "load channel -> `Channel`",
        "-> `ConstantAmplitudeSignal`",
        "-> `VariableAmplitudeSignal`",
    ):
        assert line in doc
    assert "Future work:" in doc
    for item in ("PSD", "repetitions", "force and moment", "spectra", "Element"):
        assert item in doc
    for item in ("N and mm", "MPa", "Voigt order", "tensor", "72", "fe_model"):
        assert item in doc


# -- Quantity / voigt_index ---------------------------------------------------


@pytest.mark.parametrize(
    ("component", "index"),
    [
        ("s11", 0),
        ("s22", 1),
        ("s33", 2),
        ("s23", 3),
        ("s13", 4),
        ("s12", 5),
        ("e11", 0),
        ("e12", 5),
        ("Fx", None),
        ("Mz", None),
    ],
)
def test_voigt_index(component: str, index: int | None) -> None:
    """Voigt positions follow fatpy.utils.voigt; forces and moments have none."""
    assert voigt_index(component) == index


def test_quantity_units() -> None:
    """Each quantity has its fixed unit, time is in seconds."""
    assert Quantity.STRESS.unit == "MPa"
    assert Quantity.STRAIN.unit == "mm/mm"
    assert Quantity.FORCE.unit == "N"
    assert Quantity.MOMENT.unit == "N*mm"
    assert loads.TIME_UNIT == "s"


# -- other error conditions ---------------------------------------------------


def test_voigt_index_unknown_component_raises() -> None:
    """A component that belongs to no quantity."""
    with pytest.raises(LoadDefinitionError, match="Unknown load component 's99'"):
        voigt_index("s99")


def test_channel_incompatible_component_raises() -> None:
    """A component that does not belong to the channel quantity."""
    with pytest.raises(LoadDefinitionError, match="'bad': 's11' is not a force"):
        Channel("bad", Quantity.FORCE, "s11", force_channel().signal)


def test_load_case_without_channel_raises() -> None:
    """A load case needs at least one channel."""
    with pytest.raises(LoadDefinitionError, match="no channel"):
        LoadCase("empty", [])


def test_load_case_duplicate_names_raises() -> None:
    """Channel names must be unique."""
    axial = build_case(1).channels[0]
    with pytest.raises(
        LoadDefinitionError, match=r"duplicate channel names \['axial'\]"
    ):
        LoadCase("dup", [axial, axial])


def test_load_case_mixed_periodic_and_variable_raises() -> None:
    """Periodic and variable-amplitude channels cannot be mixed."""
    channels = [*build_case(1).channels, force_channel()]
    with pytest.raises(LoadDefinitionError, match=r"\['axial'\].*\['Fx'\]"):
        LoadCase("mixed", channels).time_axis()


def test_load_case_different_instants_raises() -> None:
    """Variable-amplitude channels must share their time instants."""
    other = VariableAmplitudeSignal(np.zeros(5), time_step=0.2)
    channels = [force_channel(), Channel("Fy", Quantity.FORCE, "Fy", other)]
    with pytest.raises(LoadDefinitionError, match="'Fy' does not share.*'Fx'"):
        LoadCase("seq", channels).time_axis()


def test_load_case_too_long_cycle_raises() -> None:
    """Nearly incommensurate periods are refused, with advice."""
    channels = [
        Channel("a", Quantity.STRESS, "s11", ConstantAmplitudeSignal(1.0, period=1.0)),
        Channel(
            "b", Quantity.STRESS, "s12", ConstantAmplitudeSignal(1.0, period=1.001)
        ),
    ]
    with pytest.raises(LoadDefinitionError, match=r"incommensurate.*1:1\.414 -> 5:7"):
        LoadCase("long", channels).history()


def test_get_unknown_channel_raises() -> None:
    """get() of a channel that does not exist."""
    with pytest.raises(KeyError, match="bending"):
        build_case(1).history().get("bending")


def test_load_definition_error_is_value_error() -> None:
    """LoadDefinitionError can be caught as a ValueError."""
    assert issubclass(LoadDefinitionError, ValueError)


# -- equality, hashing, immutability -----------------------------------------


def test_channels_compare_and_hash() -> None:
    """Comparing and hashing channels does not raise."""
    a = build_case(1).channels[0]
    b = build_case(1).channels[0]
    force = force_channel()
    assert a == b
    assert hash(a) == hash(b)
    assert a != force
    assert len({a, b, force, force_channel()}) == 3
    assert build_case(1) == build_case(1)
    assert isinstance(hash(build_case(4)), int)


def test_dataclasses_are_immutable() -> None:
    """Channel, LoadCase, ChannelHistory and LoadHistory are frozen."""
    load_case = build_case(4)
    history: LoadHistory = load_case.history()
    for obj, attribute in (
        (load_case.channels[0], "name"),
        (load_case, "name"),
        (history.channels[0], "name"),
        (history, "load_case_name"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, attribute, "changed")
    assert isinstance(load_case.channels, tuple)
    assert DEFAULT_SAMPLES_PER_PERIOD == 72


# -- sampling: common period and extremes -------------------------------------


@pytest.mark.parametrize(
    ("periods", "expected"),
    [
        ([1.0], 1.0),
        ([0.2, 0.3], 0.6),
        ([1.0, 0.5], 1.0),
        ([1.5, 2.0], 6.0),
        ([1 / 3, 0.5], 1.0),
        ([0.1 + 0.2], 0.3),
    ],
)
def test_period_lcm(periods: list[float], expected: float) -> None:
    """The least common multiple is exact for decimal and rational periods."""
    assert period_lcm(periods) == pytest.approx(expected, rel=RTOL)


def test_period_lcm_empty_raises() -> None:
    """No period at all."""
    with pytest.raises(LoadDefinitionError, match="At least one period"):
        period_lcm([])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_period_lcm_invalid_period_raises(bad: float) -> None:
    """A zero, negative or non-finite period."""
    with pytest.raises(LoadDefinitionError, match="finite and positive"):
        period_lcm([1.0, bad])


def test_period_lcm_below_precision_raises() -> None:
    """A positive period shorter than TIME_PRECISION."""
    with pytest.raises(LoadDefinitionError, match="below TIME_PRECISION"):
        period_lcm([1e-12])


def test_default_sampling_is_72_points_per_period() -> None:
    """Default: 72 equal steps (5 deg) in the shortest period, ends included."""
    time = common_time_axis([ConstantAmplitudeSignal(1.0, frequency=2.0)])
    assert time.size == 73
    np.testing.assert_allclose(np.diff(time), 0.5 / 72, rtol=RTOL)


def test_samples_per_period_is_adjustable() -> None:
    """The user can refine or coarsen the sampling."""
    channels = build_case(1).channels
    assert len(LoadCase("coarse", channels, samples_per_period=8).history()) == 9
    assert len(LoadCase("fine", channels, samples_per_period=360).history()) == 361
    sine = ConstantAmplitudeSignal(1.0, period=1.0)
    assert common_time_axis([sine], samples_per_period=4).size == 5


def test_common_time_axis_requires_four_samples() -> None:
    """samples_per_period below 4, or fractional, is refused."""
    sine = ConstantAmplitudeSignal(1.0, period=1.0)
    with pytest.raises(LoadDefinitionError, match=r"integer >= 4, got 3"):
        common_time_axis([sine], samples_per_period=3)
    with pytest.raises(LoadDefinitionError, match=r"integer >= 4, got 4\.5"):
        common_time_axis([sine], samples_per_period=4.5)  # type: ignore[arg-type]


def test_common_time_axis_without_periodic_signal_raises() -> None:
    """At least one signal must be periodic."""
    static_signal = ConstantAmplitudeSignal(mean=1.0, waveform=Waveform.CONSTANT)
    with pytest.raises(LoadDefinitionError, match="periodic signal is needed"):
        common_time_axis([static_signal])


@pytest.mark.parametrize("waveform", [Waveform.SINE, Waveform.TRIANGLE])
@pytest.mark.parametrize("phase", [7.0, 13.3, 100.0, -45.0])
@pytest.mark.parametrize("samples", [4, 7, 72])
def test_sampling_hits_max_and_min_of_every_channel(
    waveform: Waveform, phase: float, samples: int
) -> None:
    """2 Hz and 3 Hz channels with any phase: exact maximum and minimum."""
    channels = [
        Channel(
            "a",
            Quantity.STRESS,
            "s11",
            ConstantAmplitudeSignal(200.0, 50.0, waveform, frequency=2.0, phase=0.0),
        ),
        Channel(
            "b",
            Quantity.STRESS,
            "s12",
            ConstantAmplitudeSignal(115.0, -10.0, waveform, frequency=3.0, phase=phase),
        ),
    ]
    history = LoadCase("peaks", channels, samples_per_period=samples).history()
    for name, high, low in (("a", 250.0, -150.0), ("b", 105.0, -125.0)):
        values = history.get(name).values
        assert values.max() == pytest.approx(high, rel=RTOL)
        assert values.min() == pytest.approx(low, rel=RTOL)
    assert np.all(np.diff(history.time) > 0.0)
    assert history.time[0] == 0.0
    assert history.time[-1] == pytest.approx(1.0, rel=RTOL)


def test_peaks_are_added_to_the_uniform_grid() -> None:
    """Off-grid peaks are inserted; every grid instant is kept."""
    sine = ConstantAmplitudeSignal(1.0, period=1.0, phase=7.0)
    time = common_time_axis([sine], samples_per_period=72)
    grid = np.arange(73) / 72
    assert all(np.any(np.isclose(time, t, rtol=0.0, atol=1e-12)) for t in grid)
    assert time.size == 73 + 2
    assert sine.evaluate(time).max() == pytest.approx(1.0, rel=RTOL)


def test_peaks_on_the_grid_keep_a_uniform_step() -> None:
    """Phase 0 with 72 samples: peaks are on the grid, nothing is added."""
    time = common_time_axis([ConstantAmplitudeSignal(1.0, period=1.0)])
    assert time.size == 73
    np.testing.assert_allclose(np.diff(time), 1.0 / 72, rtol=1e-6)


# -- FE results: superposition ------------------------------------------------

TENSION = (0.01, 0.0, 0.0, 0.0, 0.0, 0.0)  # MPa per N, times the node id
TORSION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.002)  # MPa per N*mm, times the node id
PRESSURE = (0.0, 0.5, 0.5, 0.0, 0.0, 0.0)  # MPa per MPa, times the node id


def unit_file(
    tmp_path: Path, name: str, stress: tuple[float, ...], **kw: object
) -> Path:
    """Single-step unit load file (time 1)."""
    table = fe_table(times=(1.0,), stress=stress, **kw)  # type: ignore[arg-type]
    return write_fe(tmp_path / f"{name}.csv", table)


def node_factor() -> NDArray[np.float64]:
    """Node ids of the sorted results, as a (3, 1) column."""
    return np.array([1.0, 2.0, 3.0])[:, np.newaxis]


def test_superpose_unit_loads_with_factors(tmp_path: Path) -> None:
    """Tension 1000 N + torsion 5e4 N*mm, static."""
    files = [
        FEResultFile(
            unit_file(tmp_path, "tension", TENSION), 1000.0, linear_elastic=True
        ),
        FEResultFile(unit_file(tmp_path, "torsion", TORSION), 5e4, linear_elastic=True),
    ]
    results = superpose(files)
    assert results.stress.shape == (3, 1, 6)
    np.testing.assert_allclose(results.stress[:, 0, 0:1], 10.0 * node_factor())
    np.testing.assert_allclose(results.stress[:, 0, 5:6], 100.0 * node_factor())
    np.testing.assert_allclose(results.elastic_strain, results.stress * 1e-5)
    np.testing.assert_array_equal(results.node_ids, [1, 2, 3])
    assert "1000 * " in results.source and "50000 * " in results.source


def test_superpose_out_of_phase_signals(tmp_path: Path) -> None:
    """Tension x sin + torsion x sin(+90 deg): 73 steps, exact peaks."""
    files = [
        FEResultFile(
            unit_file(tmp_path, "tension", TENSION),
            1000.0,
            ConstantAmplitudeSignal(1.0, frequency=1.0),
            linear_elastic=True,
        ),
        FEResultFile(
            unit_file(tmp_path, "torsion", TORSION),
            5e4,
            ConstantAmplitudeSignal(1.0, frequency=1.0, phase=90.0),
            linear_elastic=True,
        ),
    ]
    results = superpose(files)
    assert results.stress.shape == (3, 73, 6)
    t = results.time
    np.testing.assert_allclose(
        results.stress[:, :, 0], 10.0 * node_factor() * np.sin(2 * np.pi * t), atol=ATOL
    )
    np.testing.assert_allclose(
        results.stress[:, :, 5],
        100.0 * node_factor() * np.cos(2 * np.pi * t),
        atol=ATOL,
    )
    assert results.stress[2, :, 0].max() == pytest.approx(30.0, rel=RTOL)
    one = results.at_node(3)
    s11, s12 = one.stress[:, 0] / 30.0, one.stress[:, 5] / 300.0
    np.testing.assert_allclose(s11**2 + s12**2, 1.0, rtol=1e-9)


def test_superpose_with_static_preload(tmp_path: Path) -> None:
    """A single-step file without signal is added at every time step."""
    files = [
        FEResultFile(
            unit_file(tmp_path, "tension", TENSION),
            1000.0,
            ConstantAmplitudeSignal(1.0, frequency=1.0),
            linear_elastic=True,
        ),
        FEResultFile(
            unit_file(tmp_path, "pressure", PRESSURE), 4.0, linear_elastic=True
        ),
    ]
    results = superpose(files, samples_per_period=8)
    assert results.n_steps == 9
    np.testing.assert_allclose(
        results.stress[:, :, 1], 2.0 * node_factor() * np.ones(9)
    )


def test_superpose_measured_force_record(tmp_path: Path) -> None:
    """A unit load scaled by a measured record keeps the record instants."""
    record = VariableAmplitudeSignal(FX_VALUES, time_step=0.1)
    path = unit_file(tmp_path, "tension", TENSION)
    results = superpose([FEResultFile(path, 1.0, record, linear_elastic=True)])
    np.testing.assert_allclose(results.time, [0.0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(results.stress[0, :, 0], 0.01 * FX_VALUES)


def test_superpose_real_fe_results_factor_one(tmp_path: Path) -> None:
    """A real (elastic-plastic) result enters unchanged, with a preload."""
    lines = set_cell(fe_lines(fe_table()), 3, "PE11", "1e-3")
    real = write_lines(tmp_path / "real.csv", lines)
    alone = superpose([FEResultFile(real)])
    np.testing.assert_array_equal(alone.stress, read_fe_csv(real).stress)
    assert alone.plastic_strain[2, 1, 0] == 1e-3
    pressure = unit_file(tmp_path, "pressure", PRESSURE)
    both = superpose(
        [FEResultFile(real), FEResultFile(pressure, 2.0, linear_elastic=True)]
    )
    np.testing.assert_allclose(
        both.stress[:, :, 1] - alone.stress[:, :, 1], node_factor() * np.ones(3)
    )
    np.testing.assert_array_equal(both.time, TIMES)


def test_superpose_node_subset(tmp_path: Path) -> None:
    """node_ids is passed to every file."""
    path = unit_file(tmp_path, "tension", TENSION)
    results = superpose([FEResultFile(path, 2.0, linear_elastic=True)], node_ids=[2])
    np.testing.assert_array_equal(results.node_ids, [2])
    np.testing.assert_allclose(results.stress[0, 0, 0], 0.04)


def test_superpose_different_nodes_raises(tmp_path: Path) -> None:
    """Superposed files must have the same node ids."""
    files = [
        FEResultFile(unit_file(tmp_path, "tension", TENSION), 1.0, linear_elastic=True),
        FEResultFile(
            unit_file(tmp_path, "torsion", TORSION, nodes=(1, 2, 4)),
            1.0,
            linear_elastic=True,
        ),
    ]
    with pytest.raises(
        FEResultError,
        match=r"torsion\.csv and .*tension\.csv have different nodes: only in "
        r".*torsion\.csv: \[4\], only in .*tension\.csv: \[3\]",
    ):
        superpose(files)


def test_superpose_different_coordinates_raises(tmp_path: Path) -> None:
    """Superposed files must have the same coordinates."""
    lines = set_cell(fe_lines(fe_table(times=(1.0,), stress=TORSION)), 4, "z", "0.6")
    files = [
        FEResultFile(unit_file(tmp_path, "tension", TENSION), 1.0, linear_elastic=True),
        FEResultFile(write_lines(tmp_path / "t.csv", lines), 1.0, linear_elastic=True),
    ]
    with pytest.raises(FEResultError, match=r"coordinates of node 2 differ by 0\.1"):
        superpose(files)


def test_superpose_plastic_unit_load_raises(tmp_path: Path) -> None:
    """The PE check applies to every file declared linear-elastic."""
    lines = set_cell(
        fe_lines(fe_table(times=(1.0,), stress=TENSION)), 2, "PE33", "1e-6"
    )
    path = write_lines(tmp_path / "tension.csv", lines)
    with pytest.raises(PlasticStrainError, match=r"tension\.csv, line 2: .*node 3"):
        superpose([FEResultFile(path, 1000.0, linear_elastic=True)])


def test_superpose_signal_on_multi_step_file_raises(tmp_path: Path) -> None:
    """Only a single-step (unit load) file can be scaled by a signal."""
    path = write_fe(tmp_path / "steps.csv", fe_table())
    sine = ConstantAmplitudeSignal(1.0, frequency=1.0)
    with pytest.raises(FEResultError, match=r"steps\.csv: .*one time step.*got 3"):
        superpose([FEResultFile(path, 1.0, sine, linear_elastic=True)])


def test_superpose_different_time_steps_raises(tmp_path: Path) -> None:
    """Multi-step files must share their time steps."""
    files = [
        FEResultFile(write_fe(tmp_path / "a.csv", fe_table())),
        FEResultFile(write_fe(tmp_path / "b.csv", fe_table(times=(0.0, 1.0, 3.0)))),
    ]
    with pytest.raises(FEResultError, match=r"b\.csv: its 3 time steps differ"):
        superpose(files)


def test_superpose_mixed_signals_raises(tmp_path: Path) -> None:
    """Periodic and measured signals cannot be combined."""
    files = [
        FEResultFile(
            unit_file(tmp_path, "tension", TENSION),
            1.0,
            ConstantAmplitudeSignal(1.0, frequency=1.0),
            linear_elastic=True,
        ),
        FEResultFile(
            unit_file(tmp_path, "torsion", TORSION),
            1.0,
            VariableAmplitudeSignal(FX_VALUES, time_step=0.1),
            linear_elastic=True,
        ),
    ]
    with pytest.raises(LoadDefinitionError, match="Superposition mixes periodic"):
        superpose(files)


def test_superpose_failing_signal_names_file(tmp_path: Path) -> None:
    """A signal that cannot be evaluated names its file."""
    path = unit_file(tmp_path, "tension", TENSION)
    with pytest.raises(LoadDefinitionError, match=r"tension\.csv: broken signal"):
        superpose([FEResultFile(path, 1.0, FailingSignal(), linear_elastic=True)])


def test_superpose_without_file_raises() -> None:
    """At least one file."""
    with pytest.raises(LoadDefinitionError, match="At least one FE result file"):
        superpose([])


def test_scaling_requires_linear_elastic_declaration() -> None:
    """Only linear-elastic results can be scaled; the factor must be finite."""
    with pytest.raises(LoadDefinitionError, match="linear_elastic=True"):
        FEResultFile("a.csv", 2.0)
    sine = ConstantAmplitudeSignal(1.0, frequency=1.0)
    with pytest.raises(LoadDefinitionError, match="linear_elastic=True"):
        FEResultFile("a.csv", 1.0, sine)
    with pytest.raises(LoadDefinitionError, match="factor must be finite"):
        FEResultFile("a.csv", float("nan"), linear_elastic=True)
    assert FEResultFile("a.csv").factor == 1.0
