"""Test functions for damage parameters."""

import pytest
import numpy as np
# from numpy.typing import NDArray

from fatpy.core.energy_life import damage_parameters as dp


@pytest.fixture
def en_curve_parameters() -> dict[str, float]:
    """Parameters of the e-N curve in the form of Manson-Coffin and Basquin equation.

    Returns:
        dict[str, float]: Parameters including:
          fat_strength_coef: Manson-Coffin and Basquin equation fatigue
                                strength coefficient
          fat_ductility_coef: Manson-Coffin and Basquin equation fatigue
                                ductility coefficient
          fat_strength_exp: Manson-Coffin and Basquin equation fatigue
                                strength exponent
          fat_ductility_exp: Manson-Coffin and Basquin equation fatigue
                                ductility exponent
          elastic_modulus: Young's / Elastic modulus
    """
    params = {
        "fat_strength_coef": 475.4,
        "fat_ductility_coef": 0.612,
        "fat_strength_exp": -0.078,
        "fat_ductility_exp": -0.62,
        "elastic_modulus": 162000.0,
    }
    return params


@pytest.fixture
def stress_strain_values() -> dict[str, float]:
    """Stress / Strain values.

    Returns:
        dict[str, float]: Parameters including:
          strain_amp: Strain amplitude
          stress_amp: Stress amplitude
          mean_stress: Mean stress
    """
    params = {"strain_amp": 0.0135, "stress_amp": 290.0, "mean_stress": 10.0}
    return params


@pytest.fixture
def stress_strain_values_array() -> dict[str, np.ndarray]:
    """Stress / Strain values as arrays for testing swt with array inputs.

    Returns:
        dict[str, np.ndarray]: Parameters including:
          strain_amp: Strain amplitude
          stress_amp: Stress amplitude
          mean_stress: Mean stress
    """
    params = {
        "strain_amp": np.array([0.0135, 0.0135]),
        "stress_amp": np.array([290.0, 290.0]),
        "mean_stress": np.array([10.0, 10.0]),
    }
    return params


@pytest.fixture
def invalid_stress_condition_array() -> dict[str, np.ndarray]:
    """Stress / Strain values that violate SWT validity condition
    (stress_amp <= abs(mean_stress)).
    """

    params = {
        "strain_amp": np.array([0.0135, 0.0135]),
        "stress_amp": np.array([10.0, 20.0]),
        "mean_stress": np.array([10.0, 10.0]),
    }
    return params


def test_swt(
    en_curve_parameters: dict[str, float], stress_strain_values: dict[str, float]
) -> None:
    """Tests Smith-Watson-Topper (SWT) damage parameter for mean stress correction
    in strain-life.

    """

    n = dp.swt(en_curve_parameters, stress_strain_values)

    elastic_modulus: float = en_curve_parameters["elastic_modulus"]
    strain_amp: float = stress_strain_values["strain_amp"]
    mean_stress: float = stress_strain_values["mean_stress"]
    stress_amp: float = stress_strain_values["stress_amp"]
    p_swt = dp.calc_dmg_param_swt(elastic_modulus, strain_amp, mean_stress, stress_amp)

    assert n == 278
    assert p_swt == 810.0


def test_swt_array_returns_elementwise_cycles(
    en_curve_parameters: dict[str, float],
    stress_strain_values_array: dict[str, np.ndarray],
) -> None:
    """swt_array should solve SWT for each broadcasted input entry."""

    n_values = dp.swt(en_curve_parameters, stress_strain_values_array)

    assert np.array_equal(n_values, np.array([278, 278]))


def test_swt_array_raises_for_invalid_stress_condition(
    en_curve_parameters: dict[str, float],
    invalid_stress_condition_array: dict[str, np.ndarray],
) -> None:
    """SWT is not valid when stress_amp <= abs(mean_stress)."""

    with pytest.raises(ValueError, match="SWT is only valid"):
        dp.swt(en_curve_parameters, invalid_stress_condition_array)
