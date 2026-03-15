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


def test_swt(
    en_curve_parameters: dict[str, float], stress_strain_values: dict[str, float]
) -> None:
    """Tests Smith-Watson-Topper (SWT) damage parameter for mean stress correction
    in strain-life.

    """
    sig_f, eps_f, b, c, young_modulus = en_curve_parameters.values()
    eps_a, sig_a, sig_m = stress_strain_values.values()

    n = dp.swt(sig_f, b, eps_f, c, young_modulus, eps_a, sig_m, sig_a)
    p_swt = np.sqrt(young_modulus * eps_a * (sig_m + sig_a))

    assert n == 278
    assert p_swt == 810.0
