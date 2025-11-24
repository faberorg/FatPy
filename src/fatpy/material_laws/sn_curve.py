"""Stress-life curve methods of material laws.

Provides implementations of Wöhler (S-N) curve models along with methods for converting
between stress amplitude and fatigue life in both directions.
"""

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
        """
        self.SN_C = SN_C
        self.SN_w = SN_w

    def stress_amp(self, life: ArrayLike) -> ArrayLike:
        """Calculate stress amplitude from fatigue life using the Wöhler power law.

        Parameters:
        life : ArrayLike
            The fatigue life (N) in cycles.

        Returns:
        ArrayLike
            The calculated stress amplitude (σ_a) in MPa.
        """
        return np.power((self.SN_C) / np.asarray(life), 1 / self.SN_w)

    def life(self, stress_amp: ArrayLike) -> ArrayLike:
        """Calculate fatigue life from stress amplitude using the Wöhler power law.

        Parameters:
        stress_amp : ArrayLike
            The stress amplitude (σ_a) in MPa.

        Returns:
        ArrayLike
            The calculated fatigue life (N) in cycles.
        """
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
        """
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
            The fatigue life (N) in cycles.

        Returns:
        ArrayLike
            The calculated stress amplitude (σ_a) in MPa.
        """
        life_array = np.asarray(life)

        # Calculate stress amplitude
        stress_amp = self.A * np.power(
            (self.C * (life_array + self.B)) / (life_array + self.C), self.beta
        )

        return stress_amp

    def life(self, stress_amp: ArrayLike) -> ArrayLike:
        """Calculate fatigue life from stress amplitude using Newton method.

        Uses a vectorized Newton solver to find the inverse of:
        σ_a = A * (C * (N + B) / (N + C))^β

        Parameters:
        stress_amp : ArrayLike
            The stress amplitude (σ_a) in MPa.

        Returns:
        ArrayLike
            The calculated fatigue life (N) in cycles.
        """
        stress_amp_array = np.asarray(stress_amp)

        # Initialize solution array with starting guess
        N = np.full_like(stress_amp_array, 1e5, dtype=np.float64)

        # Newton-Raphson parameters
        max_iterations = 100
        tolerance = 1e-6

        for _ in range(max_iterations):
            # Calculate function value f(N) = A * (C*(N+B)/(N+C))^β - σ_a
            f_N = (
                self.A * np.power((self.C * (N + self.B)) / (N + self.C), self.beta)
                - stress_amp_array
            )

            # Calculate derivative f'(N) = A*β*C^β*(N+B)^(β-1)*(C-B)/(N+C)^(β+1)
            f_prime_N = (
                self.A
                * self.beta
                * np.power(self.C, self.beta)
                * (
                    (np.power(N + self.B, self.beta - 1) * (self.C - self.B))
                    / np.power(N + self.C, self.beta + 1)
                )
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
                break

            N = N_new

        return N
