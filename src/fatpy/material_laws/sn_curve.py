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
