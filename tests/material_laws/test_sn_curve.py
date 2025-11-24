"""Tests for S-N curve implementations.

This module tests the implementation of S-N curve models for fatigue analysis.
The tests verify:
    1. Correct mathematical implementation of the S-N curve models
    2. Inverse relationship between stress amplitude and fatigue life calculations
    3. Proper handling of different input types (scalars and arrays)
    4. Numerical accuracy and consistency
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from fatpy.material_laws.sn_curve import WholerPowerLaw


@pytest.fixture
def sn_curve_sample() -> WholerPowerLaw:
    """Fixture providing a sample Wöhler power law S-N curve model.

    Returns:
        WholerPowerLaw: S-N curve with material parameters C=2.2e13 MPa^3, w=3.
    """
    return WholerPowerLaw(SN_C=2.2e13, SN_w=3.0)


@pytest.fixture
def stress_amp_sample() -> NDArray[np.float64]:
    """Fixture providing sample stress amplitudes for testing.

    Returns:
        NDArray[np.float64]: Array of stress amplitudes in MPa.
    """
    return np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)


@pytest.fixture
def life_sample() -> NDArray[np.float64]:
    """Fixture providing sample fatigue lives corresponding to stress_amp_sample.

    Returns:
        NDArray[np.float64]: Array of fatigue lives in cycles.
    """
    # Calculated using N = C / σ_a^w with C=2.2e13, w=3
    return np.array(
        [22000000.0, 2750000.0, 814814.8, 343750.0, 176000.0], dtype=np.float64
    )


class TestWholerPowerLaw:
    """Test cases for the Wöhler power law S-N curve model."""

    def test_life_calculation_scalar(
        self,
        sn_curve_sample: WholerPowerLaw,
    ) -> None:
        """Test fatigue life calculation with scalar stress amplitude input.

        Uses the reference case: σ_a = 300 MPa should give N ≈ 814,815 cycles.
        """
        stress_amp = 300.0  # MPa
        expected_life = 814814.815  # cycles (more precise value)

        calculated_life = np.asarray(sn_curve_sample.life(stress_amp))

        np.testing.assert_allclose(calculated_life, expected_life, rtol=1e-6)

    def test_stress_amp_calculation_scalar(
        self,
        sn_curve_sample: WholerPowerLaw,
    ) -> None:
        """Test stress amplitude calculation with scalar fatigue life input.

        Uses the reference case: N ≈ 814,814 cycles should give σ_a = 300 MPa.
        """
        life = 814814  # cycles
        expected_stress_amp = 300.0  # MPa

        calculated_stress_amp = np.asarray(sn_curve_sample.stress_amp(life))

        np.testing.assert_allclose(
            calculated_stress_amp, expected_stress_amp, rtol=1e-6
        )

    def test_inverse_relationship(
        self,
        sn_curve_sample: WholerPowerLaw,
        stress_amp_sample: NDArray[np.float64],
    ) -> None:
        """Test that stress_amp and life methods are mathematically inverse.

        Verifies the relationship: stress_amp(life(σ)) = σ for all stress values.

        Args:
            sn_curve_sample: S-N curve model fixture.
            stress_amp_sample: Array of test stress amplitudes.
        """
        # Calculate life from stress, then stress from life
        calculated_lives = sn_curve_sample.life(stress_amp_sample)
        recovered_stress_amps = sn_curve_sample.stress_amp(calculated_lives)

        # Should recover original stress amplitudes exactly
        assert np.allclose(recovered_stress_amps, stress_amp_sample, rtol=1e-12)

    def test_life_calculation_array(
        self,
        sn_curve_sample: WholerPowerLaw,
        stress_amp_sample: NDArray[np.float64],
        life_sample: NDArray[np.float64],
    ) -> None:
        """Test fatigue life calculation with array stress amplitude inputs.

        Args:
            sn_curve_sample: S-N curve model fixture.
            stress_amp_sample: Array of test stress amplitudes.
            life_sample: Expected fatigue lives corresponding to stress_amp_sample.
        """
        calculated_lives = np.asarray(sn_curve_sample.life(stress_amp_sample))
        assert calculated_lives.shape == stress_amp_sample.shape
        np.testing.assert_allclose(calculated_lives, life_sample, rtol=1e-5)

    def test_stress_amp_calculation_array(
        self,
        sn_curve_sample: WholerPowerLaw,
        stress_amp_sample: NDArray[np.float64],
        life_sample: NDArray[np.float64],
    ) -> None:
        """Test stress amplitude calculation with array fatigue life inputs.

        Args:
            sn_curve_sample: S-N curve model fixture.
            stress_amp_sample: Expected stress amplitudes.
            life_sample: Array of test fatigue lives.
        """
        calculated_stress_amps = np.asarray(sn_curve_sample.stress_amp(life_sample))

        assert calculated_stress_amps.shape == life_sample.shape
        assert np.allclose(calculated_stress_amps, stress_amp_sample, rtol=1e-5)

    def test_mathematical_consistency(
        self,
        sn_curve_sample: WholerPowerLaw,
    ) -> None:
        """Test mathematical consistency of the Wöhler power law implementation.

        Verifies that the implemented formulas match the theoretical relationships:
        - σ_a = (C/N)^(1/w)
        - N = C / σ_a^w
        """
        # Test specific values with known analytical solution
        stress_amp = 300.0  # MPa
        C, w = 2.2e13, 3.0

        # Direct calculation using the mathematical formula
        expected_life = C / (stress_amp**w)
        expected_stress = np.power(C / expected_life, 1 / w)

        # Test implementation
        calculated_life = np.asarray(sn_curve_sample.life(stress_amp))
        calculated_stress = np.asarray(sn_curve_sample.stress_amp(expected_life))

        assert np.allclose(calculated_life, expected_life, rtol=1e-12)
        assert np.allclose(calculated_stress, expected_stress, rtol=1e-12)
