"""Test functions for uniaxial stress equivalent amplitude calculations.

Tests cover input validation, mathematical correctness, and edge cases for all
four equivalent stress amplitude calculation methods: SWT, Goodman, Gerber, and Morrow.
"""

from typing import Tuple

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from fatpy.core.stress_life.damage_params.uniaxial_stress_eq_amp import (
    ASME_mean_stress_correction_method,
    calc_stress_eq_amp_ASME,
    calc_stress_eq_amp_bagci,
    calc_stress_eq_amp_gerber,
    calc_stress_eq_amp_goodman,
    calc_stress_eq_amp_half_slope,
    calc_stress_eq_amp_linear,
    calc_stress_eq_amp_morrow,
    calc_stress_eq_amp_smith,
    calc_stress_eq_amp_soderberg,
    calc_stress_eq_amp_swt,
    calc_stress_eq_amp_walker,
)


@pytest.fixture
def array_inputs() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    stress_amp = np.array([150.0, 500.0, 80.0, 200.0])
    mean_stress = np.array([100.0, 150.0, 30.0, 0.0])
    return stress_amp, mean_stress


class TestASMEMeanStressCorrectionMethod:
    def test_basic_calculation(self) -> None:
        result = ASME_mean_stress_correction_method(180.0, 100.0, 500.0)
        expected = 180.0 / np.sqrt(1.0 - (100.0 / 500.0) ** 2)
        assert_allclose(result, expected)

    def test_array_inputs(
        self,
        array_inputs: Tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        stress_amp, mean_stress = array_inputs
        result = ASME_mean_stress_correction_method(stress_amp, mean_stress, 700.0)
        assert result.shape == (4,)

    def test_invalid_yield_strength(self) -> None:
        with pytest.raises(ValueError):
            for ys in [0.0, -500.0]:
                ASME_mean_stress_correction_method(100.0, 50.0, ys)

    def test_mean_stress_yield_strength_comparison_error(self) -> None:
        with pytest.raises(ValueError):
            for ms in [500.0, -500.0, 600.0, -600.0]:
                ASME_mean_stress_correction_method(100.0, ms, 500.0)

    def test_negative_mean_stress_no_correction(self) -> None:
        result = ASME_mean_stress_correction_method(
            180.0, -100.0, 500.0, allow_neg_mean_stress=False
        )
        expected = 180.0
        assert_allclose(result, expected)


class TestCalcStressEqAmpASME:
    def test_basic_calculation(self) -> None:
        result = calc_stress_eq_amp_ASME(180.0, 100.0, 500.0)
        expected = 180.0 / np.sqrt(1.0 - (100.0 / 500.0) ** 2)
        assert_allclose(result, expected)

    def test_array_inputs(
        self,
        array_inputs: Tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        stress_amp, mean_stress = array_inputs
        result = calc_stress_eq_amp_ASME(stress_amp, mean_stress, 700.0)
        assert result.shape == (4,)

    def test_invalid_yield_strength(self) -> None:
        with pytest.raises(ValueError):
            for ys in [0.0, -500.0]:
                calc_stress_eq_amp_ASME(100.0, 50.0, ys)

    def test_mean_stress_yield_strength_comparison_error(self) -> None:
        with pytest.raises(ValueError):
            for ms in [500.0, -500.0, 600.0, -600.0]:
                calc_stress_eq_amp_ASME(100.0, ms, 500.0)


class TestCalcStressEqAmpBagci:
    def test_basic_calculation(self) -> None:
        result = calc_stress_eq_amp_bagci(180.0, 100.0, 500.0)
        expected = 180.0 / (1.0 - (100.0 / 500.0) ** 4)
        assert_allclose(result, expected)

    def test_array_inputs(
        self,
        array_inputs: Tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        stress_amp, mean_stress = array_inputs
        result = calc_stress_eq_amp_bagci(stress_amp, mean_stress, 700.0)
        assert result.shape == (4,)

    def test_invalid_yield_strength(self) -> None:
        with pytest.raises(ValueError):
            for ys in [0.0, -500.0]:
                calc_stress_eq_amp_bagci(100.0, 50.0, ys)

    def test_mean_stress_yield_strength_comparison_error(self) -> None:
        with pytest.raises(ValueError):
            for ms in [500.0, -500.0]:
                calc_stress_eq_amp_bagci(100.0, ms, 500.0)

    def test_mean_stress_yield_strength_comparison_warning(self) -> None:
        with pytest.warns(UserWarning):
            for ms in [600.0, -600.0]:
                calc_stress_eq_amp_bagci(100.0, ms, 500.0)


class TestCalcStressEqAmpGerber:
    def test_basic_calculation(self) -> None:
        result = calc_stress_eq_amp_gerber(180.0, 100.0, 500.0)
        expected = 180.0 / (1.0 - (100.0 / 500.0) ** 2)
        assert_allclose(result, expected)

    def test_array_inputs(
        self,
        array_inputs: Tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        stress_amp, mean_stress = array_inputs
        result = calc_stress_eq_amp_gerber(stress_amp, mean_stress, 700.0)
        assert result.shape == (4,)

    def test_invalid_uts(self) -> None:
        with pytest.raises(ValueError):
            for uts in [0.0, -500.0]:
                calc_stress_eq_amp_gerber(100.0, 50.0, uts)

    def test_mean_stress_uts_comparison_error(self) -> None:
        with pytest.raises(ValueError):
            for ms in [500.0, -500.0]:
                calc_stress_eq_amp_gerber(100.0, ms, 500.0)

    def test_mean_stress_uts_comparison_warning(self) -> None:
        with pytest.warns(UserWarning):
            for ms in [600.0, -600.0]:
                calc_stress_eq_amp_gerber(100.0, ms, 500.0)


class TestCalcStressEqAmpGoodman:
    def test_basic_calculation(self) -> None:
        result = calc_stress_eq_amp_goodman(180.0, 100.0, 500.0)
        expected = 180.0 / (1.0 - (100.0 / 500.0))
        assert_allclose(result, expected)

    def test_array_inputs(
        self,
        array_inputs: Tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        stress_amp, mean_stress = array_inputs
        result = calc_stress_eq_amp_goodman(stress_amp, mean_stress, 700.0)
        assert result.shape == (4,)

    def test_invalid_uts(self) -> None:
        with pytest.raises(ValueError):
            for uts in [0.0, -500.0]:
                calc_stress_eq_amp_goodman(100.0, 50.0, uts)

    def test_mean_stress_uts_comparison_error(self) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_goodman(100.0, 500.0, 500.0)

    def test_mean_stress_uts_comparison_warning(self) -> None:
        with pytest.warns(UserWarning):
            calc_stress_eq_amp_goodman(100.0, 600.0, 500.0)


class TestCalcStressEqHalfSlope:
    def test_basic_calculation(self) -> None:
        result = calc_stress_eq_amp_half_slope(180.0, 100.0, 500.0)
        expected = 180.0 / (1.0 - (100.0 / (2 * 500.0)))
        assert_allclose(result, expected)

    def test_array_inputs(
        self,
        array_inputs: Tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        stress_amp, mean_stress = array_inputs
        result = calc_stress_eq_amp_half_slope(stress_amp, mean_stress, 700.0)
        assert result.shape == (4,)

    def test_invalid_uts(self) -> None:
        with pytest.raises(ValueError):
            for uts in [0.0, -500.0]:
                calc_stress_eq_amp_half_slope(100.0, 50.0, uts)

    def test_mean_stress_uts_comparison_error(self) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_half_slope(100.0, 500.0, 250.0)

    def test_mean_stress_uts_comparison_warning(self) -> None:
        with pytest.warns(UserWarning):
            calc_stress_eq_amp_half_slope(100.0, 650.0, 300.0)


class TestCalcStressEqAmpLinear:
    def test_basic_calculation(self) -> None:
        result = calc_stress_eq_amp_linear(180.0, 100.0, 500.0)
        expected = 180.0 / (1.0 - (100.0 / 500.0))
        assert_allclose(result, expected)

    def test_array_inputs(
        self,
        array_inputs: Tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        stress_amp, mean_stress = array_inputs
        result = calc_stress_eq_amp_linear(stress_amp, mean_stress, 700.0)
        assert result.shape == (4,)

    def test_invalid_material_param(self) -> None:
        with pytest.raises(ValueError):
            for mat_param in [0.0, -500.0]:
                calc_stress_eq_amp_linear(100.0, 50.0, mat_param)

    def test_mean_stress_material_param_comparison_error(self) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_linear(100.0, 500.0, 500.0)

    def test_mean_stress_material_param_comparison_warning(self) -> None:
        with pytest.warns(UserWarning):
            calc_stress_eq_amp_linear(100.0, 600.0, 500.0)


class TestCalcStressEqAmpMorrow:
    def test_basic_calculation(self) -> None:
        result = calc_stress_eq_amp_morrow(180.0, 100.0, 500.0)
        expected = 180.0 / (1.0 - (100.0 / 500.0))
        assert_allclose(result, expected)

    def test_array_inputs(
        self,
        array_inputs: Tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        stress_amp, mean_stress = array_inputs
        result = calc_stress_eq_amp_morrow(stress_amp, mean_stress, 700.0)
        assert result.shape == (4,)

    def test_invalid_true_fracture_stress(self) -> None:
        with pytest.raises(ValueError):
            for true_fracture_stress in [0.0, -500.0]:
                calc_stress_eq_amp_morrow(100.0, 50.0, true_fracture_stress)

    def test_mean_stress_true_fracture_stress_comparison_error(self) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_morrow(100.0, 500.0, 500.0)

    def test_mean_stress_true_fracture_stress_comparison_warning(self) -> None:
        with pytest.warns(UserWarning):
            calc_stress_eq_amp_morrow(100.0, 600.0, 500.0)


class TestCalcStressEqAmpSoderberg:
    def test_basic_calculation(self) -> None:
        result = calc_stress_eq_amp_soderberg(180.0, 100.0, 500.0)
        expected = 180.0 / (1.0 - (100.0 / 500.0))
        assert_allclose(result, expected)

    def test_array_inputs(
        self,
        array_inputs: Tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        stress_amp, mean_stress = array_inputs
        result = calc_stress_eq_amp_soderberg(stress_amp, mean_stress, 700.0)
        assert result.shape == (4,)

    def test_invalid_yield_strength(self) -> None:
        with pytest.raises(ValueError):
            for yield_strength in [0.0, -500.0]:
                calc_stress_eq_amp_soderberg(100.0, 50.0, yield_strength)

    def test_mean_stress_yield_strength_comparison_error(self) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_soderberg(100.0, 500.0, 500.0)

    def test_mean_stress_yield_strength_comparison_warning(self) -> None:
        with pytest.warns(UserWarning):
            calc_stress_eq_amp_soderberg(100.0, 600.0, 500.0)


class TestCalcStressEqAmpSmith:
    def test_basic_calculation(self) -> None:
        result = calc_stress_eq_amp_smith(180.0, 100.0, 500.0)
        expected = (180.0 * (1 + (100.0 / 500.0))) / (1.0 - (100.0 / 500.0))
        assert_allclose(result, expected)

    def test_array_inputs(
        self,
        array_inputs: Tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        stress_amp, mean_stress = array_inputs
        result = calc_stress_eq_amp_smith(stress_amp, mean_stress, 700.0)
        assert result.shape == (4,)

    def test_invalid_uts(self) -> None:
        with pytest.raises(ValueError):
            for uts in [0.0, -500.0]:
                calc_stress_eq_amp_smith(100.0, 50.0, uts)

    def test_mean_stress_uts_comparison_error(self) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_smith(100.0, 500.0, 500.0)

    def test_mean_stress_uts_comparison_warning(self) -> None:
        with pytest.warns(UserWarning):
            calc_stress_eq_amp_smith(100.0, 600.0, 500.0)


class TestCalcStressEqAmpSwt:
    def test_basic_calculation(self) -> None:
        for mean_stress, stress_amp in [(-100.0, 180.0), (100.0, 180.0)]:
            result = calc_stress_eq_amp_swt(stress_amp, mean_stress)
            expected = np.sqrt((stress_amp + mean_stress) * stress_amp)
        assert_allclose(result, expected)

    def test_array_inputs(
        self,
        array_inputs: Tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        stress_amp, mean_stress = array_inputs
        result = calc_stress_eq_amp_swt(stress_amp, mean_stress)
        assert result.shape == (4,)

    def test_negative_stress_amplitude(self) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_swt(-100.0, 500.0)

    def test_swt_validity_condition(self) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_swt(400.0, -500.0)


class TestCalcStressEqAmpWalker:
    def test_basic_calculation(self) -> None:
        result = calc_stress_eq_amp_walker(180.0, 100.0, 0.4)
        expected = (180.0 + 100.0) ** 0.6 * 180.0**0.4
        assert_allclose(result, expected)

    def test_array_inputs(
        self,
        array_inputs: Tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        stress_amp, mean_stress = array_inputs
        result = calc_stress_eq_amp_walker(stress_amp, mean_stress, 0.4)
        assert result.shape == (4,)

    def test_invalid_walker_parameter(self) -> None:
        with pytest.raises(ValueError):
            for walker_param in [-1.0, 2.0]:
                calc_stress_eq_amp_walker(100.0, 50.0, walker_param)
