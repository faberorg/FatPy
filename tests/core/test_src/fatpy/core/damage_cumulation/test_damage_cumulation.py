"""Test functions for damage accumulation rules."""

import pytest
import numpy as np
# from numpy.typing import NDArray

from fatpy.core.damage_cumulation import damage_cumulation_palmgren_meiner as dcpm


@pytest.fixture
def damage_cumulation_parameters() -> dict[str, float]:
    """Fixture providing parameters for damage cumulation tests.

    Returns:
        dict[str, float]: Parameters including slope_k, constant, sig_fl.
    """
    params = {
        "slope_k": 5.0,
        "constant": 1e17,  # 1e15 is on the internet
        "sig_fl": 137.97,
    }
    return params


@pytest.fixture
def fatigue_load_low() -> tuple[float, int]:
    """Fixture providing a sample fatigue load.

    Returns:
        tuple[float, int]: Sample stress and number of occurrences.
    """
    return 150.0, 5000


@pytest.fixture
def fatigue_load_hi() -> tuple[float, int]:
    """Fixture providing a sample fatigue load.

    Returns:
        tuple[float, int]: Sample stress and number of occurrences.
    """
    return 110.0, 100000


def test_damage_cumulation_elementary(
    damage_cumulation_parameters: dict[str, float],
    fatigue_load_low: tuple[float, int],
    fatigue_load_hi: tuple[float, int],
) -> None:
    """Elementary version
    the same slope k of the S-N curve below and above the fatigue limit
    """
    slope_k = damage_cumulation_parameters["slope_k"]
    constant = damage_cumulation_parameters["constant"]
    sig_low, n_low = fatigue_load_low
    sig_hi, n_hi = fatigue_load_hi

    d_low = dcpm.damage_cumulation_elementary(slope_k, constant, sig_low, n_low)
    d_hi = dcpm.damage_cumulation_elementary(slope_k, constant, sig_hi, n_hi)

    assert np.around(d_low, decimals=4) == 0.0038
    assert np.around(d_hi, decimals=4) == 0.0161


def test_damage_cumulation_basic(
    damage_cumulation_parameters: dict[str, float],
    fatigue_load_low: tuple[float, int],
    fatigue_load_hi: tuple[float, int],
) -> None:
    """Basic version
    the S-N curve gets horizontal at the fatigue limit,
    no damage for stresses beneath
    """
    slope_k = damage_cumulation_parameters["slope_k"]
    constant = damage_cumulation_parameters["constant"]
    sig_fl = damage_cumulation_parameters["sig_fl"]
    sig_low, n_low = fatigue_load_low
    sig_hi, n_hi = fatigue_load_hi

    d_low = dcpm.damage_cumulation_basic(slope_k, constant, sig_fl, sig_low, n_low)
    d_hi = dcpm.damage_cumulation_basic(slope_k, constant, sig_fl, sig_hi, n_hi)

    assert np.around(d_low, decimals=4) == 0.0038
    assert d_hi == 0.0


def test_damage_cumulation_haibach(
    damage_cumulation_parameters: dict[str, float],
    fatigue_load_low: tuple[float, int],
    fatigue_load_hi: tuple[float, int],
) -> None:
    """Haibach version
    the original slope_k is modified below fatigue limit to 2*slope_k-1
    """
    slope_k = damage_cumulation_parameters["slope_k"]
    constant = damage_cumulation_parameters["constant"]
    sig_fl = damage_cumulation_parameters["sig_fl"]
    sig_low, n_low = fatigue_load_low
    sig_hi, n_hi = fatigue_load_hi

    d_low = dcpm.damage_cumulation_haibach(slope_k, constant, sig_fl, sig_low, n_low)
    d_hi = dcpm.damage_cumulation_haibach(slope_k, constant, sig_fl, sig_hi, n_hi)

    assert np.around(d_low, decimals=4) == 0.0038
    assert np.around(d_hi, decimals=5) == 0.00651
