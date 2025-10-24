"""Damage parameters calculation methods for the stress-life.

"""

import numpy as np
import pytest
from numpy.typing import NDArray
from fatpy.core.stress_life.demage_parameters import s_eqa_asme

def test_s_eqa_asme() -> None:
    fat_strength_coef = np.array([475.4])
    fat_strength_exp = np.array([-0.078])
    yilel_stress = np.array([500.0])
    stress_amp = np.array([180.0])
    mean_stress = np.array([100.0])

    N = s_eqa_asme(fat_strength_coef, fat_strength_exp, yilel_stress, stress_amp, mean_stress)
    print(N)
    assert N == np.array([98366])