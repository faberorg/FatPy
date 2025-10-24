"""Damage parameters calculation methods for the stress-life."""

import numpy as np
from fatpy.core.stress_life.demage_parameters import calc_stress_eq_amp_asme


def test_calc_stress_eq_amp_asme() -> None:
    yield_stress = 500.0
    stress_amp = 180.0
    mean_stress = 100.0

    sigma_aeq = calc_stress_eq_amp_asme(yield_stress, stress_amp, mean_stress)
    assert np.around(sigma_aeq, 1) == 183.7
