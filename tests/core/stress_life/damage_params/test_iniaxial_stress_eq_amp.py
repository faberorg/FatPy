"""Test functions for uniaxial stress equivalent amplitude calculations.

Tests cover input validation, mathematical correctness, and edge cases for all
four equivalent stress amplitude calculation methods: SWT, Goodman, Gerber, and Morrow.
"""

from typing import Union

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from fatpy.core.stress_life.damage_params.uniaxial_stress_eq_amp import (
    _validate_stress_inputs,
    calc_stress_eq_amp_gerber,
    calc_stress_eq_amp_goodman,
    calc_stress_eq_amp_morrow,
    calc_stress_eq_amp_swt,
)

# Type alias for stress calculation functions
callable = Union[
    type(calc_stress_eq_amp_swt),
    type(calc_stress_eq_amp_goodman),
    type(calc_stress_eq_amp_gerber),
    type(calc_stress_eq_amp_morrow),
]


@pytest.fixture
def sample_stress_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fixture providing sample stress amplitude and mean stress data.

    Returns:
        tuple: (stress_amplitudes, mean_stresses) arrays for testing
    """
    stress_amplitudes = np.array([150.0, 500.0, 80.0])
    mean_stresses = np.array([100.0, 30.0, 0.0])
    return stress_amplitudes, mean_stresses


@pytest.fixture
def material_properties() -> dict[str, float]:
    """Fixture providing sample material properties.

    Returns:
        dict: Material properties for testing
    """
    return {
        "ult_stress": 700.0,
        "true_fract_stress": 770.0,
    }


@pytest.fixture
def zero_mean_stress_case() -> tuple[float, float]:
    """Fixture providing stress case with zero mean stress (purely alternating).

    Returns:
        tuple: (stress_amplitude, mean_stress) with mean_stress = 0
    """
    return 100.0, 0.0


@pytest.fixture
def negative_mean_stress_case() -> tuple[float, float]:
    """Fixture providing stress case with negative mean stress.

    Returns:
        tuple: (stress_amplitude, mean_stress) with negative mean_stress
    """
    return 150.0, -50.0


@pytest.fixture
def validation_test_cases() -> dict[str, tuple[float, float, float, str]]:
    """Fixture providing test cases for input validation.

    Returns:
        dict: Test cases with (stress_amp, mean_stress, material_param, param_name)
    """
    return {
        "valid_case": (100.0, 50.0, 400.0, "test parameter"),
        "negative_stress_amp": (-50.0, 30.0, 400.0, "test parameter"),
        "negative_material_param": (100.0, 50.0, -400.0, "test parameter"),
        "zero_material_param": (100.0, 50.0, 0.0, "ultimate tensile strength"),
        "mean_exceeds_material": (100.0, 450.0, 400.0, "ultimate tensile strength"),
        "mean_equals_material": (100.0, 400.0, 400.0, "ultimate tensile strength"),
    }


def test_validate_stress_inputs_valid_no_material_param(
    sample_stress_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Test validation with valid inputs and no material parameter."""
    stress_amp, mean_stress = sample_stress_data

    stress_amp_arr, mean_stress_arr = _validate_stress_inputs(stress_amp, mean_stress)

    assert_allclose(stress_amp_arr, stress_amp)
    assert_allclose(mean_stress_arr, mean_stress)
    assert stress_amp_arr.dtype == np.float64
    assert mean_stress_arr.dtype == np.float64


@pytest.mark.parametrize(
    "case_name,should_pass,expected_error",
    [
        ("valid_case", True, None),
        ("negative_stress_amp", False, "Stress amplitude must be non-negative"),
        ("negative_material_param", False, "test parameter must be positive"),
        ("zero_material_param", False, "ultimate tensile strength must be positive"),
        ("mean_exceeds_material", False, "Mean stress magnitude.*exceeds or equals"),
        ("mean_equals_material", False, "Mean stress magnitude.*exceeds or equals"),
    ],
)
def test_validate_stress_inputs_parametrized(
    validation_test_cases: dict[str, tuple[float, float, float, str]],
    case_name: str,
    should_pass: bool,
    expected_error: str | None,
) -> None:
    """Parametrized test for input validation with various cases."""
    test_case = validation_test_cases[case_name]
    stress_amp, mean_stress, material_param, param_name = test_case

    if should_pass:
        stress_amp_arr, mean_stress_arr = _validate_stress_inputs(
            stress_amp, mean_stress, material_param, param_name
        )
        assert stress_amp_arr == stress_amp
        assert mean_stress_arr == mean_stress
    else:
        with pytest.raises(ValueError, match=expected_error):
            _validate_stress_inputs(stress_amp, mean_stress, material_param, param_name)


def test_validate_stress_inputs_array_broadcasting() -> None:
    """Test validation with different array shapes."""
    stress_amp = np.array([100.0, 200.0, 150.0])
    mean_stress = 50.0
    material_param = 500.0

    stress_amp_arr, mean_stress_arr = _validate_stress_inputs(
        stress_amp, mean_stress, material_param
    )

    assert stress_amp_arr.shape == (3,)
    assert mean_stress_arr.shape == ()
    assert_allclose(stress_amp_arr, [100.0, 200.0, 150.0])
    assert mean_stress_arr == 50.0


@pytest.mark.parametrize(
    "method,stress_amp,mean_stress,material_param,expected_result",
    [
        (calc_stress_eq_amp_swt, 290.0, 10.0, None, 294.958),
        (calc_stress_eq_amp_goodman, 180.0, 100.0, 700.0, 210.0),
        (calc_stress_eq_amp_gerber, 180.0, 100.0, 700.0, 183.8),
        (calc_stress_eq_amp_morrow, 180.0, 100.0, 770.0, 206.9),
    ],
)
def test_calc_stress_eq_amp_basic_calculations(
    method: callable,
    stress_amp: float,
    mean_stress: float,
    material_param: float | None,
    expected_result: float,
) -> None:
    """Test basic calculations for all equivalent stress amplitude methods."""
    if material_param is None:
        result = method(stress_amp, mean_stress)
    else:
        result = method(stress_amp, mean_stress, material_param)

    assert_allclose(result, expected_result, rtol=1e-2)


@pytest.mark.parametrize(
    "method,material_param_key",
    [
        (calc_stress_eq_amp_swt, None),
        (calc_stress_eq_amp_goodman, "ult_stress"),
        (calc_stress_eq_amp_gerber, "ult_stress"),
        (calc_stress_eq_amp_morrow, "true_fract_stress"),
    ],
)
def test_calc_stress_eq_amp_array_inputs(
    method: callable,
    material_param_key: str | None,
    sample_stress_data: tuple[NDArray[np.float64], NDArray[np.float64]],
    material_properties: dict[str, float],
) -> None:
    """Test all methods with array inputs."""
    stress_amp, mean_stress = sample_stress_data

    if material_param_key is None:
        result = method(stress_amp, mean_stress)
        expected = np.sqrt(stress_amp * (mean_stress + stress_amp))
    else:
        material_param = material_properties[material_param_key]
        result = method(stress_amp, mean_stress, material_param)

        if method == calc_stress_eq_amp_gerber:
            expected = stress_amp / (1 - (mean_stress / material_param) ** 2)
        else:  # Goodman or Morrow
            expected = stress_amp / (1 - mean_stress / material_param)

    assert_allclose(result, expected)
    assert result.shape == (3,)


@pytest.mark.parametrize(
    "method,material_param",
    [
        (calc_stress_eq_amp_swt, None),
        (calc_stress_eq_amp_goodman, 500.0),
        (calc_stress_eq_amp_gerber, 500.0),
        (calc_stress_eq_amp_morrow, 800.0),
    ],
)
def test_calc_stress_eq_amp_zero_mean_stress(
    method: callable,
    material_param: float | None,
    zero_mean_stress_case: tuple[float, float],
) -> None:
    """Test all methods with zero mean stress (should equal stress amplitude)."""
    stress_amp, mean_stress = zero_mean_stress_case

    if material_param is None:
        result = method(stress_amp, mean_stress)
    else:
        result = method(stress_amp, mean_stress, material_param)

    assert_allclose(result, stress_amp)


def test_calc_stress_eq_amp_swt_negative_mean_stress(
    negative_mean_stress_case: tuple[float, float],
) -> None:
    """Test SWT with negative mean stress."""
    stress_amp, mean_stress = negative_mean_stress_case
    result = calc_stress_eq_amp_swt(stress_amp, mean_stress)
    expected = np.sqrt(stress_amp * (stress_amp + mean_stress))
    assert_allclose(result, expected)


def test_calc_stress_eq_amp_swt_validity_condition_violation() -> None:
    """Test that SWT validity condition violation raises ValueError."""
    with pytest.raises(
        ValueError, match="Smith-Watson-Topper parameter validity condition"
    ):
        calc_stress_eq_amp_swt(stress_amp=50.0, mean_stress=100.0)

    with pytest.raises(
        ValueError, match="Smith-Watson-Topper parameter validity condition"
    ):
        calc_stress_eq_amp_swt(stress_amp=50.0, mean_stress=-100.0)


def test_calc_stress_eq_amp_swt_validity_condition_boundary() -> None:
    """Test SWT validity condition at boundary."""
    with pytest.raises(
        ValueError, match="Smith-Watson-Topper parameter validity condition"
    ):
        calc_stress_eq_amp_swt(stress_amp=100.0, mean_stress=100.0)
