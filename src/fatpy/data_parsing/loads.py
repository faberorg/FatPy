r"""Load data parsing module.

Definition of the loading conditions applied to a component, following the
load definition of PragTic. A load is defined in one of two ways:

A) by functions (nominal stresses or strains, no FE model), see `Channel`
   and `LoadCase`;
B) by FE results exported to the FatPy CSV format, read by
   `fatpy.data_parsing.fe_model.read_fe_csv` and combined by `superpose`.

Both give stress and strain histories in Voigt form, ready for the multiaxial
criteria: ``(n_steps, 6)`` for A, ``(n_nodes, n_steps, 6)`` with the node ids
and coordinates for B (``(n_steps, 6)`` per node with
`fatpy.data_parsing.fe_model.NodalResults.at_node`).

Unit system:
    FatPy works in N and mm: forces in N, moments in N*mm, stresses in MPa,
    strains in mm/mm, coordinates in mm, time in s (or load steps). Values
    must be given in these units; nothing is converted.

Conventions:
    - Stress and strain components follow the Voigt order of
      `fatpy.utils.voigt`: (11, 22, 33, 23, 13, 12).
    - Shear strains are **tensor** components, ε_12 = γ_12 / 2, as in
      `fatpy.utils.voigt` (the Voigt vector is copied unchanged into the
      symmetric tensor) and `fatpy.struct_mech.strain` (von Mises strain
      written with ``6 (ε_12² + ε_23² + ε_13²)``).
    - Force and moment channels are external loads; they have no Voigt
      position.

A) Function definition:
    - A `Channel` gives one component of one physical `Quantity` (stress,
      strain, force or moment) a unitless time `fatpy.utils.signal.Signal`
      (sine, triangle, square, constant, or a measured record read with
      `fatpy.utils.signal.VariableAmplitudeSignal.from_csv`).
    - A `LoadCase` groups the channels acting together. Its time axis is
      either:
        - one common period of the periodic (constant-amplitude) channels,
          see `common_time_axis`, or
        - the time instants shared by the variable-amplitude channels.
      Static (``CONSTANT`` waveform) channels can be added to both.
    - `LoadCase.history` samples every channel on that time axis and returns
      a `LoadHistory`, which `LoadHistory.to_voigt_stress` and
      `LoadHistory.to_voigt_strain` turn into ``(n, 6)`` Voigt arrays.
    - A load case is a load sequence (`LoadCase.is_sequence`) when one of its
      variable-amplitude channels has no time scale.

Sampling of periodic channels:
    - The time axis spans one common period, the least common multiple of
      the periods (`period_lcm`), both ends included.
    - The shortest period is divided into `DEFAULT_SAMPLES_PER_PERIOD` (72,
      i.e. 5° steps) equal time steps; set ``samples_per_period`` to change
      it (at least `MIN_SAMPLES_PER_PERIOD`).
    - The instants of the maximum and minimum of every sine and triangle
      channel (`fatpy.utils.signal.ConstantAmplitudeSignal.peak_instants`)
      are added to the uniform grid, so the extremes are always sampled
      exactly, whatever the phases and the frequency ratios. The time step
      is then no longer uniform.

B) FE results:
    Reading FE results in the FatPy CSV format is done by
    `fatpy.data_parsing.fe_model` (format, checks, `NodalResults`). This
    module combines them with the loading: `superpose` multiplies each
    `FEResultFile` by its factor and optionally by a `Signal`, and sums them:

    - real FE results (any material model, elastic-plastic included):
      factor = 1, used as they are;
    - results under a unit load (1 N, 1 N*mm, 1 MPa): factor = real load.
      Such a file must be declared ``linear_elastic=True``; its plastic
      strains are then checked to be zero (|PE| <= 1e-12).

PragTic mapping:
    - load regime -> `LoadCase`
    - load channel -> `Channel`
    - load defined by a mathematical formula -> `ConstantAmplitudeSignal`
    - load read from a file -> `VariableAmplitudeSignal`
    - FE results under unit load, scaled and summed -> `superpose`

Future work:
    - Element and integration point results in the CSV format.
    - PSD loading (a new `Signal` class, no change to `Channel` or
      `LoadCase`).
    - Loading sequences with repetitions of load cases.
    - Conversion of force and moment channels to stresses.
    - Load spectra.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fatpy.data_parsing.fe_model import (
    FEResultError,
    NodalResults,
    check_same_nodes,
    read_fe_csv,
)
from fatpy.utils import voigt
from fatpy.utils.signal import (
    TIME_PRECISION,
    ConstantAmplitudeSignal,
    Signal,
    SignalError,
    VariableAmplitudeSignal,
)

#: Unit of time.
TIME_UNIT = "s"
#: Default number of time steps in the shortest period of a load case.
DEFAULT_SAMPLES_PER_PERIOD = 72
#: Minimum number of time steps in the shortest period.
MIN_SAMPLES_PER_PERIOD = 4
#: Maximum length of the common period, as a number of shortest periods.
MAX_CYCLE_IN_SHORTEST_PERIODS = 1000
_TIME_DIGITS = 9  # TIME_PRECISION = 10**-_TIME_DIGITS


class LoadDefinitionError(ValueError):
    """Raised when a load is inconsistently defined."""


class Quantity(Enum):
    """Physical quantity carried by a channel."""

    STRESS = "stress"
    STRAIN = "strain"
    FORCE = "force"
    MOMENT = "moment"

    @property
    def components(self) -> tuple[str, ...]:
        """Component names allowed for this quantity."""
        return _COMPONENTS[self]

    @property
    def unit(self) -> str:
        """Unit the values of this quantity are given in."""
        return _UNITS[self]


_VOIGT_SUFFIXES = ("11", "22", "33", "23", "13", "12")  # order of fatpy.utils.voigt
_COMPONENTS: dict[Quantity, tuple[str, ...]] = {
    Quantity.STRESS: tuple(f"s{s}" for s in _VOIGT_SUFFIXES),
    Quantity.STRAIN: tuple(f"e{s}" for s in _VOIGT_SUFFIXES),
    Quantity.FORCE: ("Fx", "Fy", "Fz"),
    Quantity.MOMENT: ("Mx", "My", "Mz"),
}
_UNITS: dict[Quantity, str] = {
    Quantity.STRESS: "MPa",
    Quantity.STRAIN: "mm/mm",
    Quantity.FORCE: "N",
    Quantity.MOMENT: "N*mm",
}
_VOIGT_INDEX: dict[str, int] = {
    c: i for q in (Quantity.STRESS, Quantity.STRAIN) for i, c in enumerate(q.components)
}


def voigt_index(component: str) -> int | None:
    """Position of a stress or strain component in a Voigt vector.

    Args:
        component: Component name (e.g. ``s12``, ``e23``, ``Fx``).

    Returns:
        0..5 for the 11, 22, 33, 23, 13, 12 components, ``None`` for force
        and moment components.

    Raises:
        LoadDefinitionError: If `component` belongs to no quantity.

    Example:
        >>> voigt_index("s12"), voigt_index("Fx")
        (5, None)
    """
    if component in _VOIGT_INDEX:
        return _VOIGT_INDEX[component]
    if component in Quantity.FORCE.components + Quantity.MOMENT.components:
        return None
    raise LoadDefinitionError(f"Unknown load component {component!r}")


def _read_only(values: ArrayLike) -> NDArray[np.float64]:
    """Read-only float64 copy of `values`."""
    array = np.array(values, dtype=np.float64)
    array.setflags(write=False)
    return array


# -- sampling of periodic signals ---------------------------------------------


def _period_fraction(period: float) -> Fraction:
    """Simplest fraction within `TIME_PRECISION` of `period`.

    Denominators 1, 10, ..., 1e9 are tried in turn; the last one always
    matches, since it is a rounding to 1e-9.
    """
    exact = Fraction(period)
    for digits in range(_TIME_DIGITS + 1):
        candidate = exact.limit_denominator(10**digits)
        if abs(candidate - exact) <= TIME_PRECISION:
            break
    return candidate


def period_lcm(periods: Sequence[float]) -> float:
    """Least common multiple of periods (the common period of the signals).

    Each period is replaced by the simplest `fractions.Fraction` within
    `TIME_PRECISION` (1e-9 s), so the result is exact for decimal and
    rational periods.

    Args:
        periods: Periods [s].

    Returns:
        The least common multiple [s].

    Raises:
        LoadDefinitionError: If `periods` is empty or a period is not finite
            and positive at `TIME_PRECISION`.

    Example:
        >>> period_lcm([0.2, 0.3]), period_lcm([1 / 3, 0.5])
        (0.6, 1.0)
    """
    if not periods:
        raise LoadDefinitionError("At least one period is needed")
    fractions = []
    for period in periods:
        if not math.isfinite(period) or period <= 0.0:
            raise LoadDefinitionError(
                f"Periods must be finite and positive, got {period!r}"
            )
        fraction = _period_fraction(period)
        if fraction == 0:
            raise LoadDefinitionError(
                f"Period {period!r} s is below TIME_PRECISION ({TIME_PRECISION} s)"
            )
        fractions.append(fraction)
    numerator = math.lcm(*(f.numerator for f in fractions))
    denominator = math.gcd(*(f.denominator for f in fractions))
    return numerator / denominator


def _check_samples_per_period(samples_per_period: int, context: str) -> None:
    """Raise unless `samples_per_period` is an integer >= 4."""
    if (
        not isinstance(samples_per_period, (int, np.integer))
        or samples_per_period < MIN_SAMPLES_PER_PERIOD
    ):
        raise LoadDefinitionError(
            f"{context}samples_per_period must be an integer >= "
            f"{MIN_SAMPLES_PER_PERIOD}, got {samples_per_period!r}"
        )


def common_time_axis(
    signals: Sequence[Signal],
    samples_per_period: int = DEFAULT_SAMPLES_PER_PERIOD,
    max_cycles: float = MAX_CYCLE_IN_SHORTEST_PERIODS,
) -> NDArray[np.float64]:
    """Time instants covering one common period of periodic signals.

    The axis spans the least common multiple of the periods, both ends
    included, with `samples_per_period` equal steps in the shortest period.
    The instants of the maximum and minimum of every sine and triangle signal
    are added, so the extremes are sampled exactly. Non-periodic signals are
    ignored.

    Args:
        signals: Signals; at least one must be periodic.
        samples_per_period: Number of time steps in the shortest period, an
            integer, at least `MIN_SAMPLES_PER_PERIOD`. Default 72 (5° steps).
        max_cycles: Maximum length of the common period, as a number of
            shortest periods.

    Returns:
        Sorted array of shape (n,), starting at 0 and ending at the common
        period. Without peaks off the grid, ``n = cycle / dt + 1``.

    Raises:
        LoadDefinitionError: If `samples_per_period` is too low, if no signal
            is periodic, or if the common period exceeds `max_cycles`
            shortest periods (nearly incommensurate periods).

    Example:
        >>> from fatpy.utils.signal import ConstantAmplitudeSignal
        >>> sine = ConstantAmplitudeSignal(1.0, frequency=1.0)
        >>> common_time_axis([sine], samples_per_period=4).tolist()
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    _check_samples_per_period(samples_per_period, "")
    periods = [p for s in signals if (p := s.effective_period) is not None]
    if not periods:
        raise LoadDefinitionError("At least one periodic signal is needed")
    cycle = period_lcm(periods)
    shortest = min(periods)
    if cycle > max_cycles * shortest:
        raise LoadDefinitionError(
            f"The common period of {periods} s is {cycle:g} s, more than "
            f"{max_cycles:g} times the shortest period: the periods are nearly "
            "incommensurate. Round the frequency ratio to a ratio of small "
            "integers (e.g. 1:1.414 -> 5:7)."
        )
    dt = shortest / samples_per_period
    n = round(cycle / dt) + 1
    grid = np.arange(n, dtype=np.float64) * dt
    peaks = [
        s.peak_instants(cycle)
        for s in signals
        if isinstance(s, ConstantAmplitudeSignal)
    ]
    time = np.sort(np.concatenate([*peaks, grid]))
    # Peaks come first among equal instants (stable sort): keep the exact one.
    keep = np.concatenate(([True], np.diff(time) > TIME_PRECISION))
    time = time[keep]
    time[-1] = grid[-1]  # a peak at the end is the end of the cycle
    return time


def _signals_time_axis(
    named: Sequence[tuple[str, Signal]], samples_per_period: int, context: str
) -> NDArray[np.float64]:
    """Common time axis of named signals (see `LoadCase.time_axis`)."""
    sampled = [
        (name, instants) for name, s in named if (instants := s.instants) is not None
    ]
    periodic = [name for name, s in named if s.effective_period is not None]
    if sampled and periodic:
        raise LoadDefinitionError(
            f"{context} mixes periodic channels {periodic} with "
            f"variable-amplitude channels {[n for n, _ in sampled]}"
        )
    if sampled:
        first_name, time = sampled[0]
        for name, instants in sampled[1:]:
            if instants.shape != time.shape or not np.allclose(
                instants, time, rtol=0.0, atol=TIME_PRECISION
            ):
                raise LoadDefinitionError(
                    f"{context}: channel {name!r} does not share the time "
                    f"instants of channel {first_name!r}"
                )
        return time.copy()
    if periodic:
        return common_time_axis([s for _, s in named], samples_per_period)
    return np.zeros(1)


# -- A) function definition ---------------------------------------------------


@dataclass(frozen=True)
class Channel:
    """One loaded component and its time signal.

    Attributes:
        name: Channel name, not empty and unique within a load case.
        quantity: Physical quantity of the channel.
        component: Component of `quantity` (one of `Quantity.components`).
        signal: Time signal of the channel, in the unit of `quantity`.
    """

    name: str
    quantity: Quantity
    component: str
    signal: Signal

    def __post_init__(self) -> None:
        """Check the name and that the component belongs to the quantity.

        Raises:
            LoadDefinitionError: If `name` is empty or `component` is not a
                component of `quantity`.
        """
        if not self.name.strip():
            raise LoadDefinitionError(
                f"Channel name must not be empty, got {self.name!r}"
            )
        if self.component not in self.quantity.components:
            raise LoadDefinitionError(
                f"Channel {self.name!r}: {self.component!r} is not a "
                f"{self.quantity.value} component {self.quantity.components}"
            )

    @property
    def unit(self) -> str:
        """Unit of the channel values."""
        return self.quantity.unit

    def evaluate(self, time: ArrayLike) -> NDArray[np.float64]:
        """Channel values at the given time instants [s]."""
        return self.signal.evaluate(time)


@dataclass(frozen=True, eq=False)
class ChannelHistory:
    """Values of one channel sampled on the time axis of a `LoadHistory`.

    Attributes:
        name: Channel name.
        quantity: Physical quantity of the channel.
        component: Component of `quantity`.
        unit: Unit of `values`.
        values: Sampled values, shape (n,), read-only.
    """

    name: str
    quantity: Quantity
    component: str
    unit: str
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Store `values` as a read-only float64 array."""
        object.__setattr__(self, "values", _read_only(self.values))


@dataclass(frozen=True, eq=False)
class LoadHistory:
    """Sampled load case: the value vs time of every channel.

    Attributes:
        time: Time instants [s], shape (n,), read-only.
        channels: Sampled channels, in the order of the load case.
        load_case_name: Name of the load case the history comes from.
        is_sequence: ``True`` for a load sequence (values without a time
            scale, `time` holds the indices 0, 1, 2, ...).
    """

    time: NDArray[np.float64]
    channels: tuple[ChannelHistory, ...]
    load_case_name: str
    is_sequence: bool

    def __post_init__(self) -> None:
        """Store `time` as a read-only float64 array and check the lengths.

        Raises:
            LoadDefinitionError: If a channel does not have one value per
                time instant.
        """
        object.__setattr__(self, "time", _read_only(self.time))
        object.__setattr__(self, "channels", tuple(self.channels))
        for channel in self.channels:
            if channel.values.shape != self.time.shape:
                raise LoadDefinitionError(
                    f"Channel {channel.name!r} has {channel.values.shape} values "
                    f"for {self.time.shape} time instants"
                )

    def __len__(self) -> int:
        """Number of time instants."""
        return int(self.time.size)

    def get(self, name: str) -> ChannelHistory:
        """Sampled channel by name.

        Args:
            name: Channel name.

        Returns:
            The matching `ChannelHistory`.

        Raises:
            KeyError: If there is no channel called `name`.
        """
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise KeyError(f"No channel {name!r} in {[c.name for c in self.channels]}")

    def values_array(self) -> NDArray[np.float64]:
        """Channel values as columns, shape (n, n_channels)."""
        return np.column_stack([c.values for c in self.channels])

    def to_table(self) -> dict[str, NDArray[np.float64]]:
        """Columns keyed by ``"time [s]"`` and ``"<name> [<unit>]"``.

        Returns a plain dict (FatPy does not depend on pandas types); it can
        be passed directly to ``pandas.DataFrame``.
        """
        table = {f"time [{TIME_UNIT}]": self.time}
        table.update({f"{c.name} [{c.unit}]": c.values for c in self.channels})
        return table

    def to_voigt_stress(self) -> NDArray[np.float64]:
        """Stress history in Voigt notation, as used by FatPy methods.

        Each stress channel fills its Voigt column; missing components stay
        zero.

        Returns:
            Array of shape (n, 6) [MPa].

        Raises:
            LoadDefinitionError: If a channel is not a stress, or if two
                channels give the same component.
        """
        return self._to_voigt(Quantity.STRESS)

    def to_voigt_strain(self) -> NDArray[np.float64]:
        """Strain history in Voigt notation, as used by FatPy methods.

        Each strain channel fills its Voigt column (e11 -> 0, ..., e12 -> 5);
        missing components stay zero. Shear values are tensor strains
        (ε_12 = γ_12 / 2), the convention of `fatpy.struct_mech.strain`.

        Returns:
            Array of shape (n, 6) [mm/mm].

        Raises:
            LoadDefinitionError: If a channel is not a strain, or if two
                channels give the same component.
        """
        return self._to_voigt(Quantity.STRAIN)

    def _to_voigt(self, quantity: Quantity) -> NDArray[np.float64]:
        """Fill a (n, 6) Voigt array from channels that must all be `quantity`."""
        array = np.zeros((len(self), voigt.VOIGT_COMPONENTS_COUNT))
        source: dict[int, str] = {}
        for channel in self.channels:
            if channel.quantity is not quantity:
                raise LoadDefinitionError(
                    f"Channel {channel.name!r} is {channel.quantity.value} "
                    f"({channel.component!r}), it cannot enter a Voigt "
                    f"{quantity.value} array"
                )
            index = _VOIGT_INDEX[channel.component]
            if index in source:
                raise LoadDefinitionError(
                    f"Component {channel.component!r} is given by both channels "
                    f"{source[index]!r} and {channel.name!r}"
                )
            source[index] = channel.name
            array[:, index] = channel.values
        return array


@dataclass(frozen=True)
class LoadCase:
    """Channels acting together on the component (a PragTic load regime).

    Attributes:
        name: Load case name, not empty.
        channels: Channels of the load case (stored as a tuple).
        samples_per_period: Number of time steps in the shortest period of
            the periodic channels, an integer, at least
            `MIN_SAMPLES_PER_PERIOD`. Default 72 (5° steps).
    """

    name: str
    channels: Sequence[Channel]
    samples_per_period: int = DEFAULT_SAMPLES_PER_PERIOD

    def __post_init__(self) -> None:
        """Store the channels as a tuple and validate the load case.

        Raises:
            LoadDefinitionError: If the name is empty, if there is no
                channel, if channel names are not unique, or if
                `samples_per_period` is not an integer >= 4.
        """
        object.__setattr__(self, "channels", tuple(self.channels))
        if not self.name.strip():
            raise LoadDefinitionError(
                f"Load case name must not be empty, got {self.name!r}"
            )
        if not self.channels:
            raise LoadDefinitionError(f"Load case {self.name!r} has no channel")
        names = [c.name for c in self.channels]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise LoadDefinitionError(
                f"Load case {self.name!r}: duplicate channel names {duplicates}"
            )
        _check_samples_per_period(self.samples_per_period, f"Load case {self.name!r}: ")

    @property
    def is_sequence(self) -> bool:
        """``True`` if a variable-amplitude channel has no time scale."""
        return any(
            isinstance(c.signal, VariableAmplitudeSignal) and c.signal.is_sequence
            for c in self.channels
        )

    def time_axis(self) -> NDArray[np.float64]:
        """Common time instants of all channels.

        - Periodic channels: one common period with the extremes of every
          channel, see `common_time_axis`.
        - Channels with their own instants (variable amplitude): their shared
          instants. Static (``CONSTANT``) channels may be added; they keep
          their value at every instant.
        - Only static channels: the single instant ``[0.0]``.

        Returns:
            Array of shape (n,) [s].

        Raises:
            LoadDefinitionError: If periodic channels are mixed with channels
                that have their own instants, if the latter do not share the
                same instants, or if the common period of the periodic
                channels is too long (see `MAX_CYCLE_IN_SHORTEST_PERIODS`).
        """
        return _signals_time_axis(
            [(c.name, c.signal) for c in self.channels],
            self.samples_per_period,
            f"Load case {self.name!r}",
        )

    def history(self) -> LoadHistory:
        """Sample every channel on the common time axis.

        Returns:
            The sampled `LoadHistory`.

        Raises:
            LoadDefinitionError: See `time_axis`, or if a channel signal
                cannot be evaluated (the error names the channel).
        """
        time = self.time_axis()
        channels = []
        for c in self.channels:
            try:
                values = c.evaluate(time)
            except SignalError as err:
                raise LoadDefinitionError(
                    f"Load case {self.name!r}, channel {c.name!r}: {err}"
                ) from err
            channels.append(
                ChannelHistory(c.name, c.quantity, c.component, c.unit, values)
            )
        return LoadHistory(time, tuple(channels), self.name, self.is_sequence)


@dataclass(frozen=True)
class FEResultFile:
    """One FE result file and how it enters a superposition.

    Attributes:
        path: CSV file in the FatPy format.
        factor: Multiplication factor: 1 for real FE results, the real load
            for results under a unit load (1 N, 1 N*mm, 1 MPa).
        signal: Optional time signal multiplying the results (unitless), for
            a single-step unit load result, e.g. a sine for cyclic tension.
        linear_elastic: ``True`` for a linear-elastic result (unit load): its
            plastic strains are checked to be zero (see
            `fatpy.data_parsing.fe_model.read_fe_csv`).
            Required whenever the results are scaled (`factor` != 1 or a
            `signal`), since only linear results can be scaled.
    """

    path: str | Path
    factor: float = 1.0
    signal: Signal | None = None
    linear_elastic: bool = False

    def __post_init__(self) -> None:
        """Validate the factor and the scaling.

        Raises:
            LoadDefinitionError: If `factor` is not finite, or if the results
                are scaled without being declared linear-elastic.
        """
        if not math.isfinite(self.factor):
            raise LoadDefinitionError(
                f"{self.path}: factor must be finite, got {self.factor!r}"
            )
        if (self.factor != 1.0 or self.signal is not None) and not self.linear_elastic:
            raise LoadDefinitionError(
                f"{self.path}: scaling by a factor or a signal is only valid for "
                "linear-elastic results; declare the file linear_elastic=True "
                "(real elastic-plastic results are used with factor = 1)"
            )


def superpose(
    files: Sequence[FEResultFile],
    node_ids: Sequence[int] | None = None,
    samples_per_period: int = DEFAULT_SAMPLES_PER_PERIOD,
) -> NodalResults:
    """Scale FE result files and sum them (linear superposition).

    Each file contributes ``factor * signal(t) * results``. The time axis of
    the sum is:

    - with signals: the common time axis of the signals (see
      `LoadCase.time_axis`); files scaled by a signal must have one time
      step (a unit load result);
    - otherwise the time steps of the multi-step files, which must all be
      the same.

    Files with one time step and no signal are static loads (e.g. a
    preload): they are added at every time step. Elastic strain, plastic
    strain and stress are all scaled and summed.

    Args:
        files: Result files, at least one, with the same node ids and
            coordinates.
        node_ids: Nodes to load, all if ``None``.
        samples_per_period: Time steps in the shortest period of periodic
            signals (default 72).

    Returns:
        The summed `NodalResults`.

    Raises:
        LoadDefinitionError: If `files` is empty, or the signals have no
            common time axis.
        FEResultError: See `fatpy.data_parsing.fe_model.read_fe_csv`, or if
            the files have different
            nodes, coordinates or time steps, or a file scaled by a signal
            has several time steps.
        PlasticStrainError: See `fatpy.data_parsing.fe_model.read_fe_csv`.
    """
    if not files:
        raise LoadDefinitionError("At least one FE result file is needed")
    results = [read_fe_csv(f.path, node_ids, f.linear_elastic) for f in files]
    reference = results[0]
    for other in results[1:]:
        check_same_nodes(reference, other)

    signals = [(str(f.path), f.signal) for f in files if f.signal is not None]
    for f, result in zip(files, results, strict=True):
        if f.signal is not None and result.n_steps != 1:
            raise FEResultError(
                f"{f.path}: a file scaled by a signal must have one time step "
                f"(a unit load result), got {result.n_steps}"
            )
    stepped = [r for f, r in zip(files, results, strict=True) if r.n_steps > 1]
    if signals:
        time = _signals_time_axis(signals, samples_per_period, "Superposition")
    elif stepped:
        time = stepped[0].time
    else:
        time = reference.time
    for result in stepped:
        if result.time.shape != time.shape or not np.allclose(
            result.time, time, rtol=0.0, atol=TIME_PRECISION
        ):
            raise FEResultError(
                f"{result.source}: its {result.n_steps} time steps differ from "
                f"the {time.size} time instants of the superposition"
            )

    shape = (reference.n_nodes, time.size, voigt.VOIGT_COMPONENTS_COUNT)
    sums = {name: np.zeros(shape) for name in ("elastic", "plastic", "stress")}
    for f, result in zip(files, results, strict=True):
        scale = np.full(time.size, f.factor)
        if f.signal is not None:
            try:
                scale = scale * f.signal.evaluate(time)
            except SignalError as err:
                raise LoadDefinitionError(f"{f.path}: {err}") from err
        weight = scale[np.newaxis, :, np.newaxis]
        sums["elastic"] += weight * result.elastic_strain
        sums["plastic"] += weight * result.plastic_strain
        sums["stress"] += weight * result.stress

    return NodalResults(
        node_ids=reference.node_ids,
        coordinates=reference.coordinates,
        time=time,
        elastic_strain=sums["elastic"],
        plastic_strain=sums["plastic"],
        stress=sums["stress"],
        source=" + ".join(f"{f.factor:g} * {f.path}" for f in files),
    )
