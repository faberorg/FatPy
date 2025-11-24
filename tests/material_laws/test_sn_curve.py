"""Tests for S-N curve implementations.

This module tests the implementation of S-N curve models for fatigue analysis.
The tests verify:
    1. Correct mathematical implementation of the S-N curve models
    2. Inverse relationship between stress amplitude and fatigue life calculations
    3. Proper handling of different input types (scalars and arrays)
    4. Parameter and input validation
    5. Numerical accuracy and edge case behavior
"""

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from fatpy.material_laws.sn_curve import WholerPowerLaw, WohlerKohoutVechet


@pytest.fixture
def sn_curve_sample() -> WholerPowerLaw:
    """Fixture providing a sample Wöhler power law S-N curve model.

    Returns:
        WholerPowerLaw: S-N curve with material parameters C=2.2e13 MPa^3, w=3.
    """
    return WholerPowerLaw(SN_C=2.2e13, SN_w=3.0)


@pytest.fixture
def sn_curve_negative_constants() -> dict:
    """Fixture providing negative constants for WholerPowerLaw parameter validation.

    Returns:
        dict: Dictionary with negative parameter combinations for testing validation.
    """
    return {
        "negative_C": {"SN_C": -2.2e13, "SN_w": 3.0},
        "negative_w": {"SN_C": 2.2e13, "SN_w": -3.0},
        "both_negative": {"SN_C": -2.2e13, "SN_w": -3.0},
        "zero_C": {"SN_C": 0.0, "SN_w": 3.0},
        "zero_w": {"SN_C": 2.2e13, "SN_w": 0.0},
    }


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


@pytest.fixture
def kohout_vechet_sample() -> WohlerKohoutVechet:
    """Fixture providing a sample Kohout-Věchet S-N curve model.

    Returns:
        WohlerKohoutVechet: S-N curve with validation material parameters.
    """
    return WohlerKohoutVechet(A=16651.6, B=7214.0, C=960478.0, beta=-0.351)


@pytest.fixture
def kohout_negative_constants() -> dict:
    """Fixture providing negative constants for WohlerKohoutVechet parameter validation.

    Returns:
        dict: Dictionary with invalid parameter combinations for testing validation.
    """
    return {
        "negative_A": {"A": -16651.6, "B": 7214.0, "C": 960478.0, "beta": -0.351},
        "negative_B": {"A": 16651.6, "B": -7214.0, "C": 960478.0, "beta": -0.351},
        "negative_C": {"A": 16651.6, "B": 7214.0, "C": -960478.0, "beta": -0.351},
        "positive_beta": {"A": 16651.6, "B": 7214.0, "C": 960478.0, "beta": 0.351},
        "zero_A": {"A": 0.0, "B": 7214.0, "C": 960478.0, "beta": -0.351},
        "zero_B": {"A": 16651.6, "B": 0.0, "C": 960478.0, "beta": -0.351},
        "zero_C": {"A": 16651.6, "B": 7214.0, "C": 0.0, "beta": -0.351},
        "zero_beta": {"A": 16651.6, "B": 7214.0, "C": 960478.0, "beta": 0.0},
    }


@pytest.fixture
def invalid_input_samples() -> list[ArrayLike]:
    """Fixture providing invalid input samples for testing input validation.

    Returns:
        list[ArrayLike]: List of invalid input values (negative, zero, mixed).
    """
    return [
        np.array([-1.0, 2.0, 3.0]),  # negative value
        np.array([0.0, 2.0, 3.0]),  # zero value
        np.array([1.0, -2.0, 3.0]),  # negative in middle
        -5.0,  # scalar negative
        0.0,  # scalar zero
    ]


@pytest.fixture
def kohout_stress_amp_sample() -> NDArray[np.float64]:
    """Fixture providing validation stress amplitudes for Kohout-Věchet testing.

    Returns:
        NDArray[np.float64]: Array of stress amplitudes in MPa from validation data.
    """
    # Calculated using material constants: A=16651.6, B=7214.0, C=960478.0, beta=-0.351

    return np.array(
        [
            704.0539,  # 1000 cycles
            613.4234,  # 5000 cycles
            544.8041,  # 10000 cycles
            465.5705,  # 20000 cycles
            400.0713,  # 35360 cycles
            362.5004,  # 50000 cycles
            295.7574,  # 100000 cycles
            242.2297,  # 200000 cycles
            191.7867,  # 500000 cycles
            167.1576,  # 1000000 cycles
            140.6593,  # 5000000 cycles
            136.607,  # 10000000 cycles
            133.193,  # 50000000 cycles
        ],
        dtype=np.float64,
    )


@pytest.fixture
def kohout_life_sample() -> NDArray[np.float64]:
    """Fixture providing validation fatigue lives for Kohout-Věchet testing.

    Returns:
        NDArray[np.float64]: Array of fatigue lives in cycles from validation data.
    """
    return np.array(
        [
            1000,
            5000,
            10000,
            20000,
            35360,
            50000,
            100000,
            200000,
            500000,
            1000000,
            5000000,
            10000000,
            50000000,
        ],
        dtype=np.float64,
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


class TestWohlerKohoutVechet:
    """Test cases for the Kohout-Věchet S-N curve model."""

    def test_life_calculation_scalar(
        self,
        kohout_vechet_sample: WohlerKohoutVechet,
    ) -> None:
        """Test fatigue life calculation with scalar stress amplitude input.

        Uses the reference validation case: σ_a = 400.0713 MPa → N = 35360 cycles.
        """
        stress_amp = 400.0713  # MPa
        expected_life = 35360  # cycles

        calculated_life = np.asarray(kohout_vechet_sample.life(stress_amp))

        assert np.allclose(calculated_life, expected_life, rtol=1e-2)

    def test_stress_amp_calculation_scalar(
        self,
        kohout_vechet_sample: WohlerKohoutVechet,
    ) -> None:
        """Test stress amplitude calculation with scalar fatigue life input.

        Uses the reference validation case: N = 35360 cycles → σ_a = 400.0713 MPa.
        """
        life = 35360  # cycles
        expected_stress_amp = 400.0713  # MPa (precise validation data)

        calculated_stress_amp = np.asarray(kohout_vechet_sample.stress_amp(life))

        assert np.allclose(calculated_stress_amp, expected_stress_amp, rtol=1e-6)

    def test_inverse_relationship(
        self,
        kohout_vechet_sample: WohlerKohoutVechet,
        kohout_stress_amp_sample: NDArray[np.float64],
    ) -> None:
        """Test that stress_amp and life methods are mathematically inverse.

        Verifies the relationship: stress_amp(life(σ)) = σ for all stress values.

        Args:
            kohout_vechet_sample: Kohout-Věchet S-N curve model fixture.
            kohout_stress_amp_sample: Array of validation stress amplitudes.
        """
        # Calculate life from stress, then stress from life
        calculated_lives = np.asarray(
            kohout_vechet_sample.life(kohout_stress_amp_sample)
        )
        recovered_stress_amps = np.asarray(
            kohout_vechet_sample.stress_amp(calculated_lives)
        )

        # Should recover original stress amplitudes exactly
        assert np.allclose(recovered_stress_amps, kohout_stress_amp_sample, rtol=1e-6)

    def test_life_calculation_array(
        self,
        kohout_vechet_sample: WohlerKohoutVechet,
        kohout_stress_amp_sample: NDArray[np.float64],
        kohout_life_sample: NDArray[np.float64],
    ) -> None:
        """Test fatigue life calculation with array stress amplitude inputs.

        Args:
            kohout_vechet_sample: Kohout-Věchet S-N curve model fixture.
            kohout_stress_amp_sample: Array of validation stress amplitudes.
            kohout_life_sample: Expected fatigue lives from validation data.
        """
        calculated_lives = np.asarray(
            kohout_vechet_sample.life(kohout_stress_amp_sample)
        )

        assert calculated_lives.shape == kohout_stress_amp_sample.shape
        assert np.allclose(calculated_lives, kohout_life_sample, rtol=0.01)

    def test_stress_amp_calculation_array(
        self,
        kohout_vechet_sample: WohlerKohoutVechet,
        kohout_stress_amp_sample: NDArray[np.float64],
        kohout_life_sample: NDArray[np.float64],
    ) -> None:
        """Test stress amplitude calculation with array fatigue life inputs.

        Args:
            kohout_vechet_sample: Kohout-Věchet S-N curve model fixture.
            kohout_stress_amp_sample: Expected stress amplitudes from validation data.
            kohout_life_sample: Array of validation fatigue lives.
        """
        calculated_stress_amps = np.asarray(
            kohout_vechet_sample.stress_amp(kohout_life_sample)
        )
        assert calculated_stress_amps.shape == kohout_life_sample.shape
        assert np.allclose(calculated_stress_amps, kohout_stress_amp_sample, rtol=1e-6)

    def test_edge_cases(
        self,
        kohout_vechet_sample: WohlerKohoutVechet,
    ) -> None:
        """Test edge cases and boundary conditions for numerical stability."""
        # Test with very high stress (low life) - ensures solver handles extreme values
        high_stress = 800.0  # MPa
        calculated_life = kohout_vechet_sample.life(high_stress)
        assert np.all(calculated_life > 0)  # Should be positive

        # Test with very low stress (high life) - ensures solver handles extreme values
        low_stress = 50.0  # MPa
        calculated_life = kohout_vechet_sample.life(low_stress)
        assert np.all(calculated_life > 0)  # Should be positive


class TestParameterValidation:
    """Test parameter validation for S-N curve constructors."""

    @pytest.mark.parametrize(
        "params",
        ["negative_C", "negative_w", "both_negative", "zero_C", "zero_w"],
    )
    def test_wholer_power_law_invalid_parameters(
        self, sn_curve_negative_constants: dict, params: str
    ) -> None:
        """Test that WholerPowerLaw raises ValueError for invalid parameters.

        Args:
            sn_curve_negative_constants: Fixture with invalid parameter combinations.
            params: Parameter combination key to test.
        """
        with pytest.raises(ValueError, match=r".* must be positive"):
            WholerPowerLaw(**sn_curve_negative_constants[params])

    @pytest.mark.parametrize(
        "params",
        [
            "negative_A",
            "negative_B",
            "negative_C",
            "positive_beta",
            "zero_A",
            "zero_B",
            "zero_C",
            "zero_beta",
        ],
    )
    def test_kohout_vechet_invalid_parameters(
        self, kohout_negative_constants: dict, params: str
    ) -> None:
        """Test that WohlerKohoutVechet raises ValueError for invalid parameters.

        Args:
            kohout_negative_constants: Fixture with invalid parameter combinations.
            params: Parameter combination key to test.
        """
        with pytest.raises(ValueError, match=r".* must be (positive|negative)"):
            WohlerKohoutVechet(**kohout_negative_constants[params])


class TestInputValidation:
    """Test input validation for S-N curve method parameters."""

    @pytest.mark.parametrize(
        "model_fixture,method_name",
        [
            ("sn_curve_sample", "stress_amp"),
            ("sn_curve_sample", "life"),
            ("kohout_vechet_sample", "stress_amp"),
            ("kohout_vechet_sample", "life"),
        ],
    )
    def test_invalid_input_validation(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
        method_name: str,
        invalid_input_samples: list[ArrayLike],
    ) -> None:
        """Test that all S-N curve methods raise ValueError for invalid inputs.

        Args:
            request: Pytest fixture request for dynamic fixture access.
            model_fixture: Name of the S-N curve model fixture.
            method_name: Name of the method to test ('stress_amp' or 'life').
            invalid_input_samples: List of invalid input samples to test.
        """
        model = request.getfixturevalue(model_fixture)
        method = getattr(model, method_name)

        expected_message = (
            "life must contain only positive values"
            if method_name == "stress_amp"
            else "stress_amp must contain only positive values"
        )

        for invalid_input in invalid_input_samples:
            with pytest.raises(ValueError, match=expected_message):
                method(invalid_input)
