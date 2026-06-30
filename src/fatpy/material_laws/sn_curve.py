"""Stress-life curve methods of material laws.

Provides implementations of Wöhler (S-N) curve models along with methods for converting
between stress amplitude and fatigue life in both directions.
"""

import warnings
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

def _power_law():
    pass


class SN_Curve(ABC):
    """Abstract base class for stress-life (S-N) curve models.
    
    Parameters:
    - data_points
    - log N - standard deviation
    - log S - standard deviation
    - T - T-score/ratio

    """

    def __init__(self) -> None:
        self._data_points = np.asarray([])
        self._standard_dev_log_N = float
        self._S_log_S = float
        self._T = float
        self._SN_curve_params = dict()

    @property
    def data_points(self) -> ArrayLike:
        """Return calibration data points used by the model."""
    pass

    @property
    def S_log_N(self) -> ArrayLike:
        """Return equivalent stress values at N cycles used by the model."""
    pass

    @property.setter
    def S_log_N(self, value: float) -> None:
        # added functionality for setting the property (like value checking)
        pass


    @property
    def S_log_S(self) -> ArrayLike:
        """Return equivalent stress values at N cycles used by the model."""
    pass

    @property
    def T(self) -> ArrayLike:
        """Return equivalent stress values at N cycles used by the model."""
    pass

    @property
    @abstractmethod
    def SN_curve_params(self,param_1,param_2,probability=50):
        # self._SN_curve_params.append("probability",[param1,param2])
        # ("50",[C,w])
        # ("5",[param1,param2])
        #("95",[param1,param2])
    pass

    # Usecase:
    # sn_curve_object.get_life(stress,probability = 50) -> life at given probability

    @abstractmethod
    def get_stress(self, n_cycles: ArrayLike, probability: float = 50) -> ArrayLike:
        """Calculate stress from fatigue life.

        Parameters:
        n_cycles : ArrayLike
            The fatigue life (N) in cycles.

        Returns:
        ArrayLike
            The calculated stress amplitude (σ_a) in MPa.
        """
        pass

    @abstractmethod
    def get_life(self, stress: ArrayLike, probability:float=50) -> ArrayLike:
        """Calculate fatigue life from stress amplitude.

        Parameters:
        stress_amp : ArrayLike
            The stress (σ) in MPa.

        Returns:
        ArrayLike
            The calculated fatigue life (N) in cycles.
        """
        pass

    @abstractmethod
    def least_square_curve_fitting(self,n_iterations , eps) -> None:
        # it should update the self._SN_curve_params (or rather call the method for updating)
        pass

    @abstractmethod
    def get_params_at_probability(self, new_probability)-> None:
        # 1. check if the probability is already there
        # what is the result
        pass

    @abstractmethod
    def plot_SN_curve(self, probability=50,ranges_for_x_and_y)
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

        if power_law_coef <= 0:
            raise ValueError(f"power_law_coef must be positive, got {power_law_coef}")
        if power_law_exp <= 0:
            raise ValueError(f"power_law_exp must be positive, got {power_law_exp}")

        self.power_law_coef = power_law_coef
        self.power_law_exp = power_law_exp
        self._data_points = np.asarray([] if data_points is None else data_points)
        self._S_eqN = np.asarray([] if S_eqN is None else S_eqN)

    def stress_amp(self, n_cycles: ArrayLike) -> ArrayLike:
        """Calculate stress amplitude from fatigue life using the Wöhler power law.

        Parameters:
        n_cycles : ArrayLike
            The fatigue life (N) in cycles. Must be positive.

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

        return (self.power_law_coef / n_cycles_array) ** (1 / self.power_law_exp)

    def life(self, stress_amp: ArrayLike) -> ArrayLike:
        """Calculate fatigue life from stress amplitude using the Wöhler power law.

        Parameters:
        stress_amp : ArrayLike
            The stress amplitude (σ_a) in MPa. Must be positive.

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

        return self.SN_C / (stress_amp**self.SN_w)


class WohlerKohoutVechet(SN_Curve):
    """Wöhler S-N curve model using the Kohout-Věchet method.

    This model uses a more sophisticated relationship that accounts for
    the asymptotic behavior of S-N curves at high cycle counts.
    """

    def __init__(self, A: float, B: float, C: float, beta: float):
        """Initialize the Kohout-Věchet S-N curve model.

        Parameters:
        A : float
            Material constant representing the stress amplitude scaling factor.
        B : float
            Material constant representing the life offset parameter.
        C : float
            Material constant representing the asymptotic life parameter.
        beta : float
            Material constant representing the power law exponent.

        Raises:
        ValueError
            If A, B, C parameters are not positive or if beta is not negative.
        """
        if A <= 0:
            raise ValueError(f"A must be positive, got {A}")
        if B <= 0:
            raise ValueError(f"B must be positive, got {B}")
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        if beta >= 0:
            raise ValueError(f"beta must be negative, got {beta}")

        self.A = A
        self.B = B
        self.C = C
        self.beta = beta

    def stress_amp(self, life: ArrayLike) -> ArrayLike:
        """Calculate stress amplitude from fatigue life using Kohout-Věchet method.

        ??? abstract "Math Equations"
            Uses the forward relationship:
            σ_a = A * (C * (N + B) / (N + C))^β

        Parameters:
        life : ArrayLike
            The fatigue life (N) in cycles. Must be positive.

        Returns:
        ArrayLike
            The calculated stress amplitude (σ_a) in MPa.

        Raises:
        ValueError
            If life contains non-positive values.
        """
        life_array = np.asarray(life)
        if np.any(life_array <= 0):
            raise ValueError("life must contain only positive values")

        stress_amp = (
            self.A
            * (self.C * ((life_array + self.B) / (life_array + self.C))) ** self.beta
        )

        return stress_amp

    def life(
        self,
        stress_amp: ArrayLike,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        start_life_guess: float = 1e5,
    ) -> ArrayLike:
        """Calculate fatigue life from stress amplitude using Newton solver.

        ??? abstract "Math Equations"
            Uses a vectorized Newton solver to find the inverse of:
            σ_a = A * (C * (N + B) / (N + C))^β

        Parameters:
        stress_amp : ArrayLike
            The stress amplitude (σ_a) in MPa. Must be positive.
        max_iterations : int, optional
            Maximum number of Newton iterations. Default is 100.
        tolerance : float, optional
            Tolerance for convergence. Default is 1e-6.
        start_life_guess : float, optional
            Initial guess for the fatigue life. Default is 1e5.

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

        # Initialize solution array with starting guess
        N = np.full_like(stress_amp_array, start_life_guess, dtype=np.float64)

        # Pre-calculate constants
        derivative_constant = self.A * self.beta * (self.C**self.beta)

        converged = False
        for _ in range(max_iterations):
            # Calculate function value f(N) = A * (C*(N+B)/(N+C))^β - σ_a
            f_N = (
                self.A * (self.C * (N + self.B) / (N + self.C)) ** self.beta
                - stress_amp_array
            )

            # Calculate derivative f'(N) = A*β*C^β*(N+B)^(β-1)*(C-B)/(N+C)^(β+1)
            f_prime_N = (
                derivative_constant
                * ((N + self.B) ** (self.beta - 1))
                * (self.C - self.B)
            ) / ((N + self.C) ** (self.beta + 1))

            # Avoid division by zero
            f_prime_N = np.where(np.abs(f_prime_N) < 1e-15, 1e-15, f_prime_N)
            # TODO limits based on float type or catch the case where derivative is zero, ie. B=C should B<C checked since the KV model expects it?

            # Newton update
            N_new = N - f_N / f_prime_N

            # TODO Clamp negative values to small positive number
            N_new = np.maximum(N_new, 1.0)

            # Check convergence
            relative_change = np.abs((N_new - N) / np.maximum(N, 1e-15))
            if np.all(relative_change < tolerance):
                converged = True
                break

            N = N_new

        # Issue warning if Newton solver did not converge
        if not converged:
            warnings.warn(
                f"Newton solver did not converge after {max_iterations} "
                f"iterations. Results may be inaccurate. Consider adjusting "
                f"tolerance or max_iterations.",
                RuntimeWarning,
                stacklevel=2,
            )

        return N
