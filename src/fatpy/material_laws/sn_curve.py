"""Stress-life curve methods of material laws.

Provides implementations of Wöhler (S-N) curve models along with methods for converting
between stress amplitude and fatigue life in both directions.
"""

import warnings
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class SN_Curve(ABC):
    """Abstract base class for stress-life (S-N) curve models."""

    @abstractmethod
    def stress_amp(self, life: ArrayLike) -> ArrayLike:
        """Calculate stress amplitude from fatigue life.

        Parameters:
        life : ArrayLike
            The fatigue life (N) in cycles.

        Returns:
        ArrayLike
            The calculated stress amplitude (σ_a) in MPa.
        """
        pass

    @abstractmethod
    def life(self, stress_amp: ArrayLike) -> ArrayLike:
        """Calculate fatigue life from stress amplitude.

        Parameters:
        stress_amp : ArrayLike
            The stress amplitude (σ_a) in MPa.

        Returns:
        ArrayLike
            The calculated fatigue life (N) in cycles.
        """
        pass


class WholerPowerLaw(SN_Curve):
    """Wöhler (S-N) curve model using a power law relationship."""

    def __init__(self, SN_C: float, SN_w: float):
        """Initialize the Wöhler power law model.

        Parameters:
        SN_C : float
            Material constant representing the power law coefficient (MPa^SN_w).
        SN_w : float
            Material constant representing the power law exponent.

        Raises:
        ValueError
            If any parameter is not positive.
        """
        if SN_C <= 0:
            raise ValueError(f"SN_C must be positive, got {SN_C}")
        if SN_w <= 0:
            raise ValueError(f"SN_w must be positive, got {SN_w}")

        self.SN_C = SN_C
        self.SN_w = SN_w

    def stress_amp(self, life: ArrayLike) -> ArrayLike:
        """Calculate stress amplitude from fatigue life using the Wöhler power law.

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

        return np.power((self.SN_C) / life_array, 1 / self.SN_w)

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

        return self.SN_C / np.power(stress_amp, self.SN_w)


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

        # Calculate stress amplitude
        stress_amp = self.A * np.power(
            (self.C * (life_array + self.B)) / (life_array + self.C), self.beta
        )

        return stress_amp

    def life(self, stress_amp: ArrayLike, max_iterations: int = 100) -> ArrayLike:
        """Calculate fatigue life from stress amplitude using Newton solver.

        Uses a vectorized Newton solver to find the inverse of:
        σ_a = A * (C * (N + B) / (N + C))^β

        Parameters:
        stress_amp : ArrayLike
            The stress amplitude (σ_a) in MPa. Must be positive.
        max_iterations : int, optional
            Maximum number of Newton iterations. Default is 100.

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
        N = np.full_like(stress_amp_array, 1e5, dtype=np.float64)

        # Newton solver parameters
        tolerance = 1e-6

        # Pre-calculate constants for efficiency
        derivative_constant = self.A * self.beta * np.power(self.C, self.beta)

        converged = False
        for _ in range(max_iterations):
            # Calculate function value f(N) = A * (C*(N+B)/(N+C))^β - σ_a
            f_N = (
                self.A * np.power((self.C * (N + self.B)) / (N + self.C), self.beta)
                - stress_amp_array
            )

            # Calculate derivative f'(N) = A*β*C^β*(N+B)^(β-1)*(C-B)/(N+C)^(β+1)
            f_prime_N = derivative_constant * (
                (np.power(N + self.B, self.beta - 1) * (self.C - self.B))
                / np.power(N + self.C, self.beta + 1)
            )

            # Avoid division by zero
            f_prime_N = np.where(np.abs(f_prime_N) < 1e-15, 1e-15, f_prime_N)

            # Newton update
            N_new = N - f_N / f_prime_N

            # Clamp negative values to small positive number
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
