"""Test functions for uniaxial stress equivalent amplitude calculations.

Tests cover input validation, mathematical correctness, and edge cases for all
four equivalent stress amplitude calculation methods: SWT, Goodman, Gerber, and Morrow.
"""

from typing import Any, Callable, ClassVar

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike

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

# --- Shared test fixtures / constants ---

# Broadcasting test variables
_STRESS_AMP_1D = np.array([150.0, 500.0, 80.0, 200.0])
_MEAN_STRESS_1D = np.array([100.0, 150.0, 30.0, 0.0])
_MAT_PARAM_1D = np.array([600.0, 700.0, 800.0, 900.0])
_STRESS_AMP_2D = np.array([[100.0, 200.0], [150.0, 250.0]])
_MEAN_STRESS_2D = np.array([[0.0, 50.0], [100.0, 150.0]])
_MEAN_STRESS_2D_COMPAT = np.array([[0.0, 50.0, 75.0, 100.0], [10.0, 25.0, 30.0, 0.0]])

# Scalar inputs for calculation tests
_STRESS_AMP = 180.0
_MEAN_STRESS = 100.0
_MAT_PARAM = 500.0
_MEAN_STRESS_SWT = _MEAN_STRESS_WALKER = 80.0
_STRESS_AMP_SWT = _STRESS_AMP_WALKER = 150.0

# Walker exponent (dimensionless, ∈ [0, 1])
_WALKER_PARAM = 0.5

# --- Base class ---


class _BaseEqAmpTests:
    """Shared tests for all calc_stress_eq_amp_* methods that:
    - accept (stress_amp, mean_stress, material_param, allow_neg_mean_stress)
    - raise ValueError for invalid (≤0) material param
    - raise ValueError when |mean_stress| is close to material param (isclose)
    - raise ValueError or Warn when |mean_stress| exceeds material param
    """

    correction_method: ClassVar[Callable[..., Any]]
    expected: ClassVar[float]  # expected result for (_SA, +_SM, _MAT_PARAM)
    neg_sm_expected: ClassVar[float]  # expected result for (_SA, -_SM, _MAT_PARAM)

    # --- Correct caLculation tests ---

    # Trivial case: positive mean stress is corrected → expected result
    def test_positive_mean_stress(self) -> None:
        assert_allclose(
            self.correction_method(_STRESS_AMP, _MEAN_STRESS, _MAT_PARAM), self.expected
        )

    # Negative mean stress case:
    def test_negative_mean_stress_with_correction(self) -> None:
        assert_allclose(
            self.correction_method(_STRESS_AMP, -_MEAN_STRESS, _MAT_PARAM),
            self.neg_sm_expected,
        )

    # Zero mean stress: no correction
    def test_zero_mean_stress(self) -> None:
        assert_allclose(
            self.correction_method(_STRESS_AMP, 0.0, _MAT_PARAM), _STRESS_AMP
        )

    # allow_neg_mean_stress=False:
    def test_negative_mean_stress_no_correction(self) -> None:
        assert_allclose(
            self.correction_method(
                _STRESS_AMP, -_MEAN_STRESS, _MAT_PARAM, allow_neg_mean_stress=False
            ),
            _STRESS_AMP,
        )

    # Positive sm: always corrected regardless of the flag
    def test_positive_mean_stress_allow_neg_false(self) -> None:

        assert_allclose(
            self.correction_method(
                _STRESS_AMP, _MEAN_STRESS, _MAT_PARAM, allow_neg_mean_stress=False
            ),
            self.expected,
        )

    # Array with mixed signs: negative entries return sa, positive entries corrected
    def test_mixed_sign_array_allow_neg_false(self) -> None:
        result = self.correction_method(
            _STRESS_AMP,
            [-_MEAN_STRESS, 0.0, _MEAN_STRESS],
            _MAT_PARAM,
            allow_neg_mean_stress=False,
        )
        assert_allclose(result, [_STRESS_AMP, _STRESS_AMP, self.expected])

    # --- Broadcasting ---

    @pytest.mark.parametrize(
        "stress_amp, mean_stress, mp",
        [
            pytest.param(
                _STRESS_AMP_1D,
                100.0,
                700.0,
                id="stress_amp_1d-mean_stress_scalar-mp_scalar",
            ),
            pytest.param(
                150.0,
                _MEAN_STRESS_1D,
                700.0,
                id="stress_amp_scalar-mean_stress_1d-mp_scalar",
            ),
            pytest.param(
                150.0,
                100.0,
                _MAT_PARAM_1D,
                id="stress_amp_scalar-mean_stress_scalar-mp_1d",
            ),
            pytest.param(_STRESS_AMP_1D, _MEAN_STRESS_1D, _MAT_PARAM_1D, id="all_1d"),
            pytest.param(
                _STRESS_AMP_2D,
                _MEAN_STRESS_2D,
                700.0,
                id="stress_amp_2d-mean_stress_2d-mp_scalar",
            ),
            pytest.param(
                150.0,
                _MEAN_STRESS_2D,
                700.0,
                id="stress_amp_scalar-mean_stress_2d-mp_scalar",
            ),
            pytest.param(
                _STRESS_AMP_1D,
                _MEAN_STRESS_2D_COMPAT,
                700.0,
                id="stress_amp_1d-mean_stress_2d-mp_scalar",
            ),
        ],
    )
    def test_broadcasting(
        self,
        stress_amp: ArrayLike | np.float64,
        mean_stress: ArrayLike | np.float64,
        mp: ArrayLike | np.float64,
    ) -> None:
        result = self.correction_method(stress_amp, mean_stress, mp)
        expected_shape = np.broadcast_shapes(
            np.asarray(stress_amp).shape,
            np.asarray(mean_stress).shape,
            np.asarray(mp).shape,
        )
        assert result.shape == expected_shape
        assert result.dtype == np.float64

    # --- Input validation ---

    @pytest.mark.parametrize("mp", [0.0, -500.0])
    def test_invalid_material_param(self, mp: float) -> None:
        with pytest.raises(ValueError):
            self.correction_method(100.0, 50.0, mp)

    @pytest.mark.parametrize("ms", [500.0, -500.0, 499.97, -499.97])
    def test_mean_stress_close_to_param_error(self, ms: float) -> None:
        with pytest.raises(ValueError):
            self.correction_method(100.0, ms, 500.0)

    # Default behaviour: warn when |sm| > param.
    @pytest.mark.parametrize("ms", [600.0, -600.0])
    def test_mean_stress_exceeds_param(self, ms: float) -> None:
        with pytest.warns(UserWarning):
            self.correction_method(100.0, ms, 500.0)


# --- Concrete test classes ---


class TestCalcStressEqAmpAsme(_BaseEqAmpTests):
    correction_method = staticmethod(calc_stress_eq_amp_asme)
    expected = _STRESS_AMP / np.sqrt(1.0 - (_MEAN_STRESS / _MAT_PARAM) ** 2)
    neg_sm_expected = _STRESS_AMP / np.sqrt(1.0 - (-_MEAN_STRESS / _MAT_PARAM) ** 2)

    # Override: ASME method raises ValueError
    @pytest.mark.parametrize("ms", [600.0, -600.0])
    def test_mean_stress_exceeds_param(self, ms: float) -> None:
        with pytest.raises(ValueError):
            self.correction_method(100.0, ms, 500.0)


class TestCalcStressEqAmpBagci(_BaseEqAmpTests):
    correction_method = staticmethod(calc_stress_eq_amp_bagci)
    expected = _BAGCI_EXPECTED = _STRESS_AMP / (1.0 - (_MEAN_STRESS / _MAT_PARAM) ** 4)
    neg_sm_expected = _BAGCI_EXPECTED = _STRESS_AMP / (
        1.0 - (-_MEAN_STRESS / _MAT_PARAM) ** 4
    )


class TestCalcStressEqAmpGerber(_BaseEqAmpTests):
    correction_method = staticmethod(calc_stress_eq_amp_gerber)
    expected = _STRESS_AMP / (1.0 - (_MEAN_STRESS / _MAT_PARAM) ** 2)
    neg_sm_expected = _STRESS_AMP / (1.0 - (-_MEAN_STRESS / _MAT_PARAM) ** 2)


class TestCalcStressEqAmpGoodman(_BaseEqAmpTests):
    correction_method = staticmethod(calc_stress_eq_amp_goodman)
    expected = _STRESS_AMP / (1.0 - _MEAN_STRESS / _MAT_PARAM)
    neg_sm_expected = _STRESS_AMP / (1.0 - (-_MEAN_STRESS / _MAT_PARAM))


class TestCalcStressEqAmpHalfSlope(_BaseEqAmpTests):
    correction_method = staticmethod(calc_stress_eq_amp_half_slope)
    expected = _STRESS_AMP / (1.0 - _MEAN_STRESS / (2.0 * _MAT_PARAM))
    neg_sm_expected = _STRESS_AMP / (1.0 - (-_MEAN_STRESS) / (2.0 * _MAT_PARAM))

    # Half-slope singularity is at |sm| ≈ 2*UTS → override boundary test
    @pytest.mark.parametrize("ms", [1000.0, -1000.0, 999.97, -999.97])
    def test_mean_stress_close_to_param_error(self, ms: float) -> None:
        with pytest.raises(ValueError):
            self.correction_method(100.0, ms, 500.0)


class TestCalcStressEqAmpLinear(_BaseEqAmpTests):
    correction_method = staticmethod(calc_stress_eq_amp_linear)
    expected = _STRESS_AMP / (1.0 - _MEAN_STRESS / _MAT_PARAM)
    neg_sm_expected = _STRESS_AMP / (1.0 - (-_MEAN_STRESS / _MAT_PARAM))


class TestCalcStressEqAmpMorrow(_BaseEqAmpTests):
    correction_method = staticmethod(calc_stress_eq_amp_morrow)
    expected = _STRESS_AMP / (1.0 - _MEAN_STRESS / _MAT_PARAM)
    neg_sm_expected = _STRESS_AMP / (1.0 - (-_MEAN_STRESS / _MAT_PARAM))


class TestCalcStressEqAmpSoderberg(_BaseEqAmpTests):
    correction_method = staticmethod(calc_stress_eq_amp_soderberg)
    expected = _STRESS_AMP / (1.0 - _MEAN_STRESS / _MAT_PARAM)
    neg_sm_expected = _STRESS_AMP / (1.0 - (-_MEAN_STRESS / _MAT_PARAM))


class TestCalcStressEqAmpSmith(_BaseEqAmpTests):
    correction_method = staticmethod(calc_stress_eq_amp_smith)
    expected = (
        _STRESS_AMP
        * (1.0 + _MEAN_STRESS / _MAT_PARAM)
        / (1.0 - _MEAN_STRESS / _MAT_PARAM)
    )
    neg_sm_expected = (
        _STRESS_AMP
        * (1.0 + (-_MEAN_STRESS) / _MAT_PARAM)
        / (1.0 - (-_MEAN_STRESS) / _MAT_PARAM)
    )


# --- SWT and Walker: standalone classes (different signatures and validation) ---
_SWT_EXPECTED = np.sqrt(_STRESS_AMP_SWT * (_MEAN_STRESS_SWT + _STRESS_AMP_SWT))
_SWT_NEG_SM_EXPECTED = np.sqrt(_STRESS_AMP_SWT * (-_MEAN_STRESS_SWT + _STRESS_AMP_SWT))

_WALKER_EXPECTED = (_STRESS_AMP_WALKER + _MEAN_STRESS_WALKER) ** (
    1 - _WALKER_PARAM
) * _STRESS_AMP_WALKER**_WALKER_PARAM

_WALKER_NEG_SM_EXPECTED = (_STRESS_AMP_WALKER - _MEAN_STRESS_WALKER) ** (
    1 - _WALKER_PARAM
) * _STRESS_AMP_WALKER**_WALKER_PARAM


class TestCalcStressEqAmpSwt:
    """Tests for SWT: sqrt(sa * (sm + sa)). No material param."""

    # Trivial case
    def test_positive_mean_stress(self) -> None:
        assert_allclose(
            calc_stress_eq_amp_swt(_STRESS_AMP_SWT, _MEAN_STRESS_SWT), _SWT_EXPECTED
        )

    # Negative mean stress:
    def test_negative_mean_stress_with_correction(self) -> None:
        assert_allclose(
            calc_stress_eq_amp_swt(_STRESS_AMP_SWT, -_MEAN_STRESS_SWT),
            _SWT_NEG_SM_EXPECTED,
        )

    # Zero mean stress:
    def test_zero_mean_stress(self) -> None:
        assert_allclose(calc_stress_eq_amp_swt(_STRESS_AMP_SWT, 0.0), _STRESS_AMP_SWT)

    # Negative mean stress with allow_neg_mean_stress=False:
    def test_negative_mean_stress_no_correction(self) -> None:
        result = calc_stress_eq_amp_swt(
            _STRESS_AMP_SWT, -_MEAN_STRESS_SWT, allow_neg_mean_stress=False
        )
        assert_allclose(result, _STRESS_AMP_SWT)

    # Positive mean stress with allow_neg_mean_stress=False:
    def test_positive_mean_stress_allow_neg_false(self) -> None:
        result = calc_stress_eq_amp_swt(
            _STRESS_AMP_SWT, _MEAN_STRESS_SWT, allow_neg_mean_stress=False
        )
        assert_allclose(result, _SWT_EXPECTED)

    # Array with mixed signs: negative entries return sa, positive entries corrected
    def test_mixed_sign_array_allow_neg_false(self) -> None:
        result = calc_stress_eq_amp_swt(
            _STRESS_AMP_SWT,
            [-_MEAN_STRESS_SWT, 0.0, _MEAN_STRESS_SWT],
            allow_neg_mean_stress=False,
        )
        assert_allclose(result, [_STRESS_AMP_SWT, _STRESS_AMP_SWT, _SWT_EXPECTED])

    @pytest.mark.parametrize(
        "stress_amp, mean_stress",
        [
            pytest.param(_STRESS_AMP_1D, 100.0, id="sa_1d-sm_scalar"),
            pytest.param(150.0, _MEAN_STRESS_1D, id="sa_scalar-sm_1d"),
            pytest.param(_STRESS_AMP_1D, _MEAN_STRESS_1D, id="all_1d"),
            pytest.param(_STRESS_AMP_2D, _MEAN_STRESS_2D, id="sa_2d-sm_2d"),
            pytest.param(150.0, _MEAN_STRESS_2D, id="sa_scalar-sm_2d"),
            pytest.param(_STRESS_AMP_1D, _MEAN_STRESS_2D_COMPAT, id="sa_1d-sm_2d"),
        ],
    )
    def test_broadcasting(
        self,
        stress_amp: ArrayLike | np.float64,
        mean_stress: ArrayLike | np.float64,
    ) -> None:
        result = calc_stress_eq_amp_swt(stress_amp, mean_stress)
        expected_shape = np.broadcast_shapes(
            np.asarray(stress_amp).shape,
            np.asarray(mean_stress).shape,
        )
        assert result.shape == expected_shape
        assert result.dtype == np.float64

    @pytest.mark.parametrize("sa", [-_STRESS_AMP_SWT, -1.0, -0.001])
    def test_negative_stress_amp_raises(self, sa: float) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_swt(sa, 100.0)

    @pytest.mark.parametrize(
        "sa, sm",
        [
            pytest.param(100.0, -100.0, id="sa_plus_sm_equals_zero"),
            pytest.param(100.0, -200.0, id="sa_plus_sm_negative"),
        ],
    )
    def test_validity_condition_violated(self, sa: float, sm: float) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_swt(sa, sm)


class TestCalcStressEqAmpWalker:
    """Tests for Walker: (sa + sm)^(1-γ) * sa^γ. walker_param = γ ∈ [0, 1]."""

    # Trivial case
    def test_positive_mean_stress(self) -> None:
        assert_allclose(
            calc_stress_eq_amp_walker(
                _STRESS_AMP_WALKER, _MEAN_STRESS_WALKER, _WALKER_PARAM
            ),
            _WALKER_EXPECTED,
        )

    # Negative mean stress:
    def test_negative_mean_stress_with_correction(self) -> None:
        assert_allclose(
            calc_stress_eq_amp_walker(
                _STRESS_AMP_WALKER, -_MEAN_STRESS_WALKER, _WALKER_PARAM
            ),
            _WALKER_NEG_SM_EXPECTED,
        )

    # Zero mean stress:
    def test_zero_mean_stress(self) -> None:
        assert_allclose(
            calc_stress_eq_amp_walker(_STRESS_AMP_WALKER, 0.0, _WALKER_PARAM),
            _STRESS_AMP_WALKER,
        )

    # Negative mean stress with allow_neg_mean_stress=False:
    def test_negative_mean_stress_no_correction(self) -> None:
        result = calc_stress_eq_amp_walker(
            _STRESS_AMP_WALKER,
            -_MEAN_STRESS_WALKER,
            _WALKER_PARAM,
            allow_neg_mean_stress=False,
        )
        assert_allclose(result, _STRESS_AMP_WALKER)

    # Positive mean stress with allow_neg_mean_stress=False:
    def test_positive_mean_stress_allow_neg_false(self) -> None:
        result = calc_stress_eq_amp_walker(
            _STRESS_AMP_WALKER,
            _MEAN_STRESS_WALKER,
            _WALKER_PARAM,
            allow_neg_mean_stress=False,
        )
        assert_allclose(result, _WALKER_EXPECTED)

    # Array with mixed signs: negative entries return sa, positive entries corrected
    def test_mixed_sign_array_allow_neg_false(self) -> None:
        result = calc_stress_eq_amp_walker(
            _STRESS_AMP_WALKER,
            [-_MEAN_STRESS_WALKER, 0.0, _MEAN_STRESS_WALKER],
            _WALKER_PARAM,
            allow_neg_mean_stress=False,
        )
        assert_allclose(
            result, [_STRESS_AMP_WALKER, _STRESS_AMP_WALKER, _WALKER_EXPECTED]
        )

    @pytest.mark.parametrize(
        "stress_amp, mean_stress",
        [
            pytest.param(_STRESS_AMP_1D, 100.0, id="sa_1d-sm_scalar"),
            pytest.param(150.0, _MEAN_STRESS_1D, id="sa_scalar-sm_1d"),
            pytest.param(_STRESS_AMP_1D, _MEAN_STRESS_1D, id="all_1d"),
            pytest.param(_STRESS_AMP_2D, _MEAN_STRESS_2D, id="sa_2d-sm_2d"),
            pytest.param(150.0, _MEAN_STRESS_2D, id="sa_scalar-sm_2d"),
            pytest.param(_STRESS_AMP_1D, _MEAN_STRESS_2D_COMPAT, id="sa_1d-sm_2d"),
        ],
    )
    def test_broadcasting(
        self,
        stress_amp: ArrayLike | np.float64,
        mean_stress: ArrayLike | np.float64,
    ) -> None:
        result = calc_stress_eq_amp_walker(stress_amp, mean_stress, _WALKER_PARAM)
        expected_shape = np.broadcast_shapes(
            np.asarray(stress_amp).shape,
            np.asarray(mean_stress).shape,
        )
        assert result.shape == expected_shape
        assert result.dtype == np.float64

    @pytest.mark.parametrize("sa", [-_STRESS_AMP_WALKER, -1.0, -0.001])
    def test_negative_stress_amp_raises(self, sa: float) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_walker(sa, 100.0, _WALKER_PARAM)

    @pytest.mark.parametrize(
        "sa, sm",
        [
            pytest.param(100.0, -100.0, id="sa_plus_sm_equals_zero"),
            pytest.param(100.0, -200.0, id="sa_plus_sm_negative"),
        ],
    )
    def test_validity_condition_violated(self, sa: float, sm: float) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_walker(sa, sm, _WALKER_PARAM)

    @pytest.mark.parametrize("gamma", [-0.1, 1.1, -1.0, 2.0])
    def test_invalid_walker_param(self, gamma: float) -> None:
        with pytest.raises(ValueError):
            calc_stress_eq_amp_walker(_STRESS_AMP_WALKER, _MEAN_STRESS_WALKER, gamma)
