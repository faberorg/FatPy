"""Test functions for uniaxial stress equivalent amplitude calculations.

Tests cover input validation, mathematical correctness, and edge cases for all
four equivalent stress amplitude calculation methods: SWT, Goodman, Gerber, and Morrow.
"""

from typing import Tuple

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike, NDArray

from fatpy.core.stress_life.damage_params.uniaxial_stress_eq_amp import (
    calc_stress_eq_amp_asme,
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

# Broadcasting test variables - different shapes
_SA_1D = np.array([150.0, 500.0, 80.0, 200.0])
_SM_1D = np.array([100.0, 150.0, 30.0, 0.0])
_MAT_PARAM_1D = np.array([600.0, 700.0, 800.0, 900.0])
_SA_2D = np.array([[100.0, 200.0], [150.0, 250.0]])
_SM_2D = np.array([[0.0, 50.0], [100.0, 150.0]])
# Shape (2, 4): last dim matches _SA_1D so they broadcast to (2, 4)
_SM_2D_COMPAT = np.array([[0.0, 50.0, 75.0, 100.0], [10.0, 25.0, 30.0, 0.0]])

_ASME_EXPECTED = 180.0 / np.sqrt(1.0 - (100.0 / 500.0) ** 2)


class TestCalcStressEqAmpAsme:
    # Test numerical calculation for scalar inputs, and negative mean stress control
    @pytest.mark.parametrize(
        "sa, sm, ys, allow_neg_mean_stress, expected",
        [
            pytest.param(
                180.0,
                100.0,
                500.0,
                True,
                _ASME_EXPECTED,
                id="positive_mean_stress",
            ),
            pytest.param(
                180.0,
                -100.0,
                500.0,
                True,
                _ASME_EXPECTED,
                id="negative_mean_stress_with_correction",
            ),
            pytest.param(180.0, 0.0, 500.0, True, 180.0, id="zero_mean_stress"),
            pytest.param(
                180.0,
                -100.0,
                500.0,
                False,
                180.0,
                id="negative_mean_stress_no_correction",
            ),
            pytest.param(
                180.0,
                100.0,
                500.0,
                False,
                _ASME_EXPECTED,
                id="positive_mean_stress_allow_neg_false",
            ),
        ],
    )
    def test_calculation(
        self,
        sa: float,
        sm: float,
        ys: float,
        allow_neg_mean_stress: bool,
        expected: float,
    ) -> None:
        result = calc_stress_eq_amp_asme(
            sa, sm, ys, allow_neg_mean_stress=allow_neg_mean_stress
        )
        assert_allclose(result, expected)

    # Test broadcasting behavior for various combinations of 1D and 2D array inputs
    @pytest.mark.parametrize(
        "sa, sm, mp",
        [
            pytest.param(_SA_1D, 100.0, 700.0, id="sa_1d-sm_scalar-mp_scalar"),
            pytest.param(150.0, _SM_1D, 700.0, id="sa_scalar-sm_1d-mp_scalar"),
            pytest.param(150.0, 100.0, _MAT_PARAM_1D, id="sa_scalar-sm_scalar-mp_1d"),
            pytest.param(_SA_1D, _SM_1D, _MAT_PARAM_1D, id="all_1d"),
            pytest.param(_SA_2D, _SM_2D, 700.0, id="sa_2d-sm_2d-mp_scalar"),
            pytest.param(150.0, _SM_2D, 700.0, id="sa_scalar-sm_2d-mp_scalar"),
            pytest.param(_SA_1D, _SM_2D_COMPAT, 700.0, id="sa_1d-sm_2d-mp_scalar"),
        ],
    )
    def test_broadcasting(
        self,
        sa: ArrayLike | np.float64,
        sm: ArrayLike | np.float64,
        mp: ArrayLike | np.float64,
    ) -> None:
        result = calc_stress_eq_amp_asme(sa, sm, mp)
        expected_shape = np.broadcast_shapes(
            np.asarray(sa).shape,
            np.asarray(sm).shape,
            np.asarray(mp).shape,
        )
        assert result.shape == expected_shape

    # Test ValueError and Warning conditions
    @pytest.mark.parametrize("ys", [0.0, -500.0])
    def test_invalid_yield_strength(self, ys: float) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_asme(100.0, 50.0, ys)

    @pytest.mark.parametrize("ms", [600.0, -600.0])
    def test_mean_stress_exceeds_yield_strength_error(self, ms: float) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_asme(100.0, ms, 500.0)

    @pytest.mark.parametrize("ms", [500.0, -500.0])
    def test_mean_stress_close_to_yield_strength_error(self, ms: float) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_asme(100.0, ms, 500.0)


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
