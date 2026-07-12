"""Stress-life curve methods of material laws.

Provides implementations of Wöhler (S-N) curve models along with methods for converting
between stress amplitude and fatigue life in both directions.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, TypedDict, Unpack

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


class _PowerLawParams(TypedDict):
    """TypedDict for Wöhler power law parameters.

    In self._params[probability] the keys are "coef" and "exp"
    for the power law coefficient and exponent, respectively.
    """

    power_law_coef: float
    power_law_exp: float


class SN_Curve(ABC):
    """Abstract base class for stress-life (S-N) curve models.

    Stores experimental data (stress, life, runout) and defines the
    interface that all concrete S-N curve implementations must fulfill.

    Two usage workflows are supported:

    **1. Fit from data**
        Provide experimental data via :meth:`set_data`, call :meth:`fit` to
        estimate curve parameters, then call :meth:`get_statistics` to compute
        scatter metrics. After that, :meth:`get_stress` and :meth:`get_life`
        are available at any probability level.

    **2. Direct parameter input**
        If curve parameters and statistics are already known (e.g. from
        literature or a previous fit), supply them via :meth:`set_params`
        (at ``probability=0.5``) and :meth:`set_statistics`. This enables
        :meth:`get_stress` and :meth:`get_life` at any probability without
        needing experimental data.

    Parameters dictionary
    ---------------------
    Curve parameters are stored in ``self._params`` as a ``dict`` keyed by
    probability level (``float`` in ``[0, 1]``). The 50% parameters are the
    standard base; parameters at other probabilities are derived from them
    and cached automatically by subclass implementations of
    :meth:`get_stress` / :meth:`get_life`.

    **Linear (single-segment) power law** — flat dict with two parameters::

        {
            0.5: {"coef": 1234.0, "exp": 5.6},
            0.1: {"coef":  980.0, "exp": 5.6},   # cached on first call
            0.9: {"coef": 1456.0, "exp": 5.6},   # cached on first call
        }

    **Multilinear (piecewise) power law** — nested list of segments plus a
    knee stress for each probability level::

        {
            0.5: {
                "segments": [
                    {"coef": 1234.0, "exp":  5.6},   # low-cycle segment
                    {"coef": 8500.0, "exp": 12.0},   # high-cycle segment
                ],
                "knee_stress": 250.0,   # MPa — intersection of the two segments
                "knee_life": 1.0e5,      # cycles — intersection of the two segments
            },
            0.1: {
                "segments": [
                    {"coef":  980.0, "exp":  5.6},
                    {"coef": 6800.0, "exp": 12.0},
                ],
                "knee_stress": 220.0,   # shifts because both coefs shifted
                "knee_life": 1.0e5,      # cycles — intersection of the two segments
            },
        }

    Data
    ----
    Experimental data is optional at construction time. Use :meth:`set_data`
    to supply it with built-in validation. Runout specimens (unbroken at test
    end) are flagged with ``runout == True`` and excluded from curve fitting.

    Statistics dictionary
    ---------------------
    :meth:`get_statistics` and :meth:`set_statistics` use a ``dict`` with
    the following required keys:

    ``"distribution_name"`` : str
        Name of the assumed distribution (e.g. ``"LogNormal"``).
    ``"log_S_std"`` : float
        Standard deviation of residuals in log-stress space.
    ``"log_N_std"`` : float
        Standard deviation of residuals in log-life space.
    ``"T"`` : float
        Scatter ratio (e.g. T_N = N_10% / N_90%).
    """

    def __init__(
        self,
        stress: Optional[ArrayLike] = None,
        n_cycles: Optional[ArrayLike] = None,
        runout: Optional[ArrayLike] = None,
        params: Optional[dict[str, Any]] = None,
        statistics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize with optional experimental data and/or known parameters."""
        self._stress: np.ndarray = np.empty(0, dtype=float)
        self._n_cycles: np.ndarray = np.empty(0, dtype=float)
        self._runout: np.ndarray = np.empty(0, dtype=bool)
        self._params: dict[float, dict[str, Any]] = {}
        self._statistics: Optional[dict[str, Any]] = None
        self._fat_limit: Optional[float] = None

        if stress is not None or n_cycles is not None or runout is not None:
            _stress = [] if stress is None else stress
            _n_cycles = [] if n_cycles is None else n_cycles
            if runout is None:
                runout = np.zeros(len(np.asarray(_stress, dtype=float)), dtype=bool)
            self.set_data(_stress, _n_cycles, runout)
        if params is not None:
            self.set_params(params, probability=0.5)
        if statistics is not None:
            self.set_statistics(statistics)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def stress(self) -> np.ndarray:
        """Stored stress amplitude values (MPa)."""
        return self._stress

    @property
    def n_cycles(self) -> np.ndarray:
        """Stored fatigue life values (cycles)."""
        return self._n_cycles

    @property
    def runout(self) -> np.ndarray:
        """Boolean runout mask — ``True`` where data point is a runout."""
        return self._runout

    @property
    def params(self) -> dict[float, dict[str, Any]]:
        """Curve parameters keyed by probability level."""
        return self._params

    def set_params(self, params: dict[str, Any], probability: float = 0.5) -> None:
        """Store curve parameters for the given probability level.

        The 50% (median) parameters are the standard base and must be set
        before parameters at other probabilities can be derived. Subclass
        implementations of :meth:`get_stress` and :meth:`get_life` cache
        computed parameters here automatically.

        Parameters:
        params : dict
            Model-specific parameter dict (e.g. ``{"coef": 1234, "exp": 5.6}``).
        probability : float, optional
            Probability level in ``[0, 1]``. Default ``0.5``.
        """
        self._params[probability] = params

    def get_params_at_prob(self, probability: float = 0.5) -> Optional[dict[str, Any]]:
        """Return cached parameters for the given probability, or ``None``.

        Subclass implementations of :meth:`get_stress` and :meth:`get_life`
        should call this first and only compute parameters when the result
        is ``None``.

        Parameters:
        probability : float, optional
            Probability level in ``[0, 1]``. Default ``0.5``.
        """
        return self._params.get(probability)

    @property
    def statistics(self) -> Optional[dict[str, Any]]:
        """Last statistics dict from :meth:`get_statistics`, or ``None``."""
        return self._statistics

    _REQUIRED_STATS_KEYS = frozenset(
        {"distribution_name", "log_stress_std", "log_n_cycles_std"}
    )

    def set_statistics(self, statistics: dict[str, Any]) -> None:
        """Directly set the statistics dictionary (workflow 2: known parameters).

        Use this when statistics are already known from a previous fit or
        literature, so that :meth:`get_stress` and :meth:`get_life` can be
        evaluated at probability levels other than 0.5.

        Parameters:
        statistics : dict
            Must contain keys: ``"distribution_name"``, ``"log_stress_std"``,
            ``"log_n_cycles_std"``, ``"T"``.

        Raises:
        ValueError
            If any required key is missing.
        """
        missing = self._REQUIRED_STATS_KEYS - statistics.keys()
        if missing:
            raise ValueError(f"statistics dict is missing required keys: {missing}")
        self._statistics = statistics

    def set_data(
        self,
        stress: ArrayLike,
        n_cycles: ArrayLike,
        runout: ArrayLike,
    ) -> None:
        """Set and validate experimental data.

        Parameters
        ----------
        stress : ArrayLike
            Stress amplitude (σ_a) in MPa. All values must be positive.
        n_cycles : ArrayLike
            Fatigue life (N) in cycles. All values must be positive.
        runout : ArrayLike
            Runout indicators. Use ``True`` for runouts, ``False`` for
            failures. Integer values ``1``/``0`` are also accepted.

        Raises:
            ValueError: If any stress or n_cycles values are non-positive, if
                runout contains values outside ``{0, 1}``, or if arrays
                have different lengths.
        """
        stress_arr = np.asarray(stress, dtype=float)
        n_cycles_arr = np.asarray(n_cycles, dtype=float)
        runout_arr = np.asarray(runout)

        if np.any(stress_arr <= 0):
            raise ValueError("stress must contain only positive values")
        if np.any(n_cycles_arr <= 0):
            raise ValueError("n_cycles must contain only positive values")
        if not np.all(np.isin(runout_arr, [0, 1])):
            raise ValueError("runout must contain only 0 or 1 values")
        if not (len(stress_arr) == len(n_cycles_arr) == len(runout_arr)):
            raise ValueError("stress, n_cycles, and runout must have the same length")

        self._stress = stress_arr
        self._n_cycles = n_cycles_arr
        self._runout = runout_arr.astype(bool)

    @property
    def fat_limit(self) -> Optional[float]:
        """Return the fatigue limit (endurance limit) if known, else None.

        The fatigue limit is the stress amplitude below which the material
        can endure an infinite number of cycles without failure. It may be
        set by subclasses or by user input.
        """
        return self._fat_limit

    def set_fatigue_limit(self, fat_limit: float) -> None:
        """Set the fatigue limit (endurance limit) for the material.

        Parameters:
        fat_limit : float
            The fatigue limit stress amplitude in MPa. Must be positive.

        Raises:
        ValueError
            If fat_limit is not positive.
        """
        if fat_limit <= 0:
            raise ValueError(f"fatigue limit must be positive, got {fat_limit}")
        self._fat_limit = fat_limit

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def set_params_case_specific(self, *args: Any, **kwargs: Any) -> None:
        """Set model-specific curve parameters for the given probability level.

        Each subclass defines its own named parameters. ``probability`` is
        mandatory for all implementations and must remain a keyword argument.

        Examples — concrete signatures:
            ``WholerPowerLaw.set_params_case_specific(coef, exp, probability=0.5)``
            ``WholerBilinear.set_params_case_specific(coef_1, exp_1, coef_2, exp_2, probability=0.5)``
        """

    @abstractmethod
    def get_stress(self, n_cycles: ArrayLike, probability: float = 0.5) -> ArrayLike:
        """Calculate stress amplitude from fatigue life.

        Parameters:
        n_cycles : ArrayLike
            Fatigue life (N) in cycles.
        probability : float, optional
            Probability of failure in ``[0, 1]``. Default ``0.5`` (median).

        Returns:
        ArrayLike
            Stress amplitude (σ_a) in MPa.
        """
        pass

    @abstractmethod
    def get_n_cycles(self, stress: ArrayLike, probability: float = 0.5) -> ArrayLike:
        """Calculate fatigue life from stress amplitude.

        Parameters:
        stress : ArrayLike
            Stress amplitude (σ_a) in MPa.
        probability : float, optional
            Probability of failure in ``[0, 1]``. Default ``0.5`` (median).

        Returns:
        ArrayLike
            Fatigue life (N) in cycles.
        """
        pass

    @abstractmethod
    def fit_curve_polynom(self, deg: int = 1) -> None:
        """Fit the S-N curve model to the stored experimental data.

        Runout specimens (``self._runout == True``) must be excluded from
        the fit. Updates curve parameters in-place.
        """
        pass

    @abstractmethod
    def get_statistics(self, dist: str = "LogNormal") -> dict[str, Any]:
        """Compute and store statistical metrics of the fitted model.

        Updates ``self._statistics`` and returns it.

        Parameters:
        dist : str, optional
            Distribution assumption. Default ``"LogNormal"``.

        Returns:
        dict
            Required keys: ``"distribution_name"``, ``"log_stress_std"``,
            ``"log_n_cycles_std"``, ``"T"``.
        """
        pass

    @abstractmethod
    def plot_sn_curve(self, probability: float = 0.5) -> None:
        """Plot the S-N curve at the given probability level.

        Parameters
        ----------
        probability : float, optional
            Probability of failure in ``[0, 1]``. Default ``0.5``.
        """
        pass


class WholerPowerLaw(SN_Curve):
    """Wöhler (S-N) curve model using a power law relationship."""

    def __init__(self) -> None:
        """Initialize the Wöhler power law model.

        Parameters:
        power_law_coef : float
            Material constant representing the power law coefficient (MPa^power_law_exp).
        power_law_exp : float
            Material constant representing the power law exponent.

        Raises:
        ValueError
            If any parameter is not positive.
        """
        super().__init__()

    def set_params_case_specific(
        self, probability: float = 0.5, **kwargs: Unpack[_PowerLawParams]
    ) -> None:
        """Store Wöhler power law parameters for the given probability level.

        Parameters:
        power_law_coef : float
            Material constant representing the power law coefficient (MPa^power_law_exp).
        power_law_exp : float
            Material constant representing the power law exponent.
        probability : float, optional
            Probability level in ``[0, 1]``. Default ``0.5``.

        Raises:
        ValueError
            If any parameter is not positive.
        """
        power_law_coef = kwargs.get("power_law_coef")
        power_law_exp = kwargs.get("power_law_exp")

        if power_law_coef <= 0:
            raise ValueError(f"power_law_coef must be positive, got {power_law_coef}")
        if power_law_exp <= 0:
            raise ValueError(f"power_law_exp must be positive, got {power_law_exp}")

        self._params[probability] = {
            "power_law_coef": power_law_coef,
            "power_law_exp": power_law_exp,
        }

    def get_stress(self, n_cycles: ArrayLike, probability: float = 0.5) -> ArrayLike:
        """Calculate stress amplitude from fatigue life using the Wöhler power law.

        Parameters:
        n_cycles : ArrayLike
            The fatigue life (N) in cycles. Must be positive.
        probability : float, optional
            Probability of failure in ``[0, 1]``. Default ``0.5``.

        Returns:
        ArrayLike
            The calculated stress amplitude (σ_a) in MPa.

        Raises:
        ValueError
            If n_cycles contains non-positive values.
        """
        n_cycles_array = np.asarray(n_cycles)
        if np.any(n_cycles_array <= 0):
            raise ValueError("n_cycles must contain only positive values")

        power_law_coef = self._params[probability]["power_law_coef"]
        power_law_exp = self._params[probability]["power_law_exp"]
        stress_arr = (power_law_coef / n_cycles_array) ** (1 / power_law_exp)

        return stress_arr

    def get_n_cycles(
        self, stress_amp: ArrayLike, probability: float = 0.5
    ) -> ArrayLike:
        """Calculate fatigue life from stress amplitude using the Wöhler power law.

        Parameters:
        stress_amp : ArrayLike
            The stress amplitude (σ_a) in MPa. Must be positive.
        probability : float, optional
            Probability of failure in ``[0, 1]``. Default ``0.5``.

        Returns:
        ArrayLike
            The calculated fatigue life (N) in cycles.

        Raises:
        ValueError
            If stress_amp contains non-positive values.
        """
        stress_amp_array = np.asarray(stress_amp)
        if np.any(stress_amp_array <= 0):
            raise ValueError("stress_amp must contain only positive values")

        power_law_coef = self._params[probability]["power_law_coef"]
        power_law_exp = self._params[probability]["power_law_exp"]

        return power_law_coef / (stress_amp_array**power_law_exp)

    def fit_curve_polynom(self, deg: int = 1) -> None:
        """Fit the Wöhler power law model to calibration data points.

        Parameters:
        deg : int, optional
            Degree of the polynomial to fit. Default is 1.
        """
        log_stress = np.log10(self._stress[~self._runout])
        log_n_cycles = np.log10(self._n_cycles[~self._runout])

        # sn_fit = np.polynomial.Polynomial.fit(log_n_cycles, log_stress, deg=1)
        sn_fit = np.polynomial.Polynomial.fit(
            x=np.flip(log_stress), y=np.flip(log_n_cycles), deg=deg
        )
        intercept, slope = sn_fit.convert().coef
        exp = -slope
        coef = 10 ** (-intercept / slope)
        self.set_params_case_specific(
            probability=0.5, power_law_coef=coef, power_law_exp=exp
        )

        pass

    def get_statistics(self, dist: str = "LogNormal") -> dict[str, Any]:
        """Get a statistical summary of the fitted Wöhler power law model.

        Returns:
        dict
            A dictionary containing the fitted power law coefficient and exponent.
        """
        log_stress = np.log10(self._stress[self._runout])
        stress_pred = self.get_stress(self._n_cycles[self._runout])
        log_stress_pred = np.log10(stress_pred)

        log_n_cycles = np.log10(self._n_cycles[self._runout])
        n_cycles_pred = self.get_n_cycles(self._stress[self._runout])
        log_n_cycles_pred = np.log10(n_cycles_pred)

        if dist == "LogNormal":
            residuals_stress = log_stress - log_stress_pred
            std_stress = np.std(residuals_stress, ddof=1)

            residuals_n_cycles = log_n_cycles - log_n_cycles_pred
            std_n_cycles = np.std(residuals_n_cycles, ddof=1)

            statistical_data = {
                "distribution_name": dist,
                "log_stress_std": std_stress,
                "log_n_cycles_std": std_n_cycles,
            }
        else:
            raise ValueError(f"Unsupported distribution: {dist}")

        self.set_statistics(statistical_data)

        return statistical_data

    def plot_sn_curve(self, probability: float = 0.5) -> None:
        """Plot the S-N curve at the given probability level.

        Parameters
        ----------
        probability : float, optional
            Probability of failure in ``[0, 1]``. Default ``0.5``.
        """
        failures = ~self._runout
        runouts = self._runout

        fig, ax = plt.subplots()

        if self._n_cycles.size > 0:
            ax.scatter(
                self._n_cycles[failures],
                self._stress[failures],
                marker="o",
                label="Failures",
            )
            ax.scatter(
                self._n_cycles[runouts],
                self._stress[runouts],
                marker=">",
                facecolors="none",
                edgecolors="C1",
                linewidths=1.25,
                s=60,
                label="Runouts",
            )

        if probability in self._params:
            n_range = np.logspace(
                np.log10(max(self._n_cycles.min(), 1.0)),
                np.log10(self._n_cycles.max()),
                200,
            )
            stress_line = self.get_stress(n_range, probability=probability)
            ax.loglog(n_range, stress_line, label=f"P={probability:.0%}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Fatigue life N (cycles)")
        ax.set_ylabel("Stress amplitude σ_a (MPa)")
        ax.legend()
        plt.tight_layout()
        plt.show()


# class WohlerKohoutVechet(SN_Curve):
#     """Wöhler S-N curve model using the Kohout-Věchet method.

#     This model uses a more sophisticated relationship that accounts for
#     the asymptotic behavior of S-N curves at high cycle counts.
#     """

#     def __init__(self, A: float, B: float, C: float, beta: float):
#         """Initialize the Kohout-Věchet S-N curve model.

#         Parameters:
#         A : float
#             Material constant representing the stress amplitude scaling factor.
#         B : float
#             Material constant representing the life offset parameter.
#         C : float
#             Material constant representing the asymptotic life parameter.
#         beta : float
#             Material constant representing the power law exponent.

#         Raises:
#         ValueError
#             If A, B, C parameters are not positive or if beta is not negative.
#         """
#         if A <= 0:
#             raise ValueError(f"A must be positive, got {A}")
#         if B <= 0:
#             raise ValueError(f"B must be positive, got {B}")
#         if C <= 0:
#             raise ValueError(f"C must be positive, got {C}")
#         if beta >= 0:
#             raise ValueError(f"beta must be negative, got {beta}")

#         self.A = A
#         self.B = B
#         self.C = C
#         self.beta = beta

#     def stress_amp(self, life: ArrayLike) -> ArrayLike:
#         """Calculate stress amplitude from fatigue life using Kohout-Věchet method.

#         ??? abstract "Math Equations"
#             Uses the forward relationship:
#             σ_a = A * (C * (N + B) / (N + C))^β

#         Parameters:
#         life : ArrayLike
#             The fatigue life (N) in cycles. Must be positive.

#         Returns:
#         ArrayLike
#             The calculated stress amplitude (σ_a) in MPa.

#         Raises:
#         ValueError
#             If life contains non-positive values.
#         """
#         life_array = np.asarray(life)
#         if np.any(life_array <= 0):
#             raise ValueError("life must contain only positive values")

#         stress_amp = (
#             self.A
#             * (self.C * ((life_array + self.B) / (life_array + self.C))) ** self.beta
#         )

#         return stress_amp

#     def life(
#         self,
#         stress_amp: ArrayLike,
#         max_iterations: int = 100,
#         tolerance: float = 1e-6,
#         start_life_guess: float = 1e5,
#     ) -> ArrayLike:
#         """Calculate fatigue life from stress amplitude using Newton solver.

#         ??? abstract "Math Equations"
#             Uses a vectorized Newton solver to find the inverse of:
#             σ_a = A * (C * (N + B) / (N + C))^β

#         Parameters:
#         stress_amp : ArrayLike
#             The stress amplitude (σ_a) in MPa. Must be positive.
#         max_iterations : int, optional
#             Maximum number of Newton iterations. Default is 100.
#         tolerance : float, optional
#             Tolerance for convergence. Default is 1e-6.
#         start_life_guess : float, optional
#             Initial guess for the fatigue life. Default is 1e5.

#         Returns:
#         ArrayLike
#             The calculated fatigue life (N) in cycles.

#         Raises:
#         ValueError
#             If stress_amp contains non-positive values.
#         """
#         stress_amp_array = np.asarray(stress_amp)
#         if np.any(stress_amp_array <= 0):
#             raise ValueError("stress_amp must contain only positive values")

#         # Initialize solution array with starting guess
#         N = np.full_like(stress_amp_array, start_life_guess, dtype=np.float64)

#         # Pre-calculate constants
#         derivative_constant = self.A * self.beta * (self.C**self.beta)

#         converged = False
#         for _ in range(max_iterations):
#             # Calculate function value f(N) = A * (C*(N+B)/(N+C))^β - σ_a
#             f_N = (
#                 self.A * (self.C * (N + self.B) / (N + self.C)) ** self.beta
#                 - stress_amp_array
#             )

#             # Calculate derivative f'(N) = A*β*C^β*(N+B)^(β-1)*(C-B)/(N+C)^(β+1)
#             f_prime_N = (
#                 derivative_constant
#                 * ((N + self.B) ** (self.beta - 1))
#                 * (self.C - self.B)
#             ) / ((N + self.C) ** (self.beta + 1))

#             # Avoid division by zero
#             f_prime_N = np.where(np.abs(f_prime_N) < 1e-15, 1e-15, f_prime_N)
#             # TODO limits based on float type or catch the case where derivative is zero, ie. B=C should B<C checked since the KV model expects it?

#             # Newton update
#             N_new = N - f_N / f_prime_N

#             # TODO Clamp negative values to small positive number
#             N_new = np.maximum(N_new, 1.0)

#             # Check convergence
#             relative_change = np.abs((N_new - N) / np.maximum(N, 1e-15))
#             if np.all(relative_change < tolerance):
#                 converged = True
#                 break

#             N = N_new

#         # Issue warning if Newton solver did not converge
#         if not converged:
#             warnings.warn(
#                 f"Newton solver did not converge after {max_iterations} "
#                 f"iterations. Results may be inaccurate. Consider adjusting "
#                 f"tolerance or max_iterations.",
#                 RuntimeWarning,
#                 stacklevel=2,
#             )

#         return N
