# tests/stress_life/test_s_eqa_goodman.py
import numpy as np
import pytest

# Import the function (not the module); alias to avoid name shadowing
from fatpy.core.stress_life.damage_params.s_eqa_goodman import (
    s_eqa_goodman as goodman_fn,
)


def test_returns_tuple_and_dtypes() -> None:
    N, sigma_aeq = goodman_fn(
        stress_amp=150.0,
        mean_stress=25.0,
        fat_strength_coef=400.0,
        fat_strength_exp=-0.1,
        ult_stress=700.0,
    )
    # Accept NumPy scalars or arrays:
    N_arr = np.asarray(N)
    sig_arr = np.asarray(sigma_aeq)

    assert N_arr.dtype == np.int64
    assert sig_arr.dtype == np.float64

    # For scalar inputs, function returns 0-D; this documents the behavior.
    assert N_arr.ndim == 0
    assert sig_arr.ndim == 0


def test_example_numeric_values() -> None:
    """Concrete example to check both outputs."""
    sa = 180.0
    sm = 100.0
    sf = 475.4
    b = -0.078
    uts = 700.0

    N, sigma_aeq = goodman_fn(
        stress_amp=sa,
        mean_stress=sm,
        fat_strength_coef=sf,
        fat_strength_exp=b,
        ult_stress=uts,
    )

    # sigma_aeq = 180 / (1 - 100/700) = 210
    assert np.isclose(float(sigma_aeq), 210.0, rtol=0.0, atol=1e-12)
    # N ≈ 17709 after rounding
    assert int(N) == 17709


def test_vectorization_and_broadcast() -> None:
    sa = np.array([150.0, 180.0, 210.0])
    sm = np.array([0.0, 50.0, 100.0])

    N, sigma_aeq = goodman_fn(
        stress_amp=sa,
        mean_stress=sm,
        fat_strength_coef=475.4,
        fat_strength_exp=-0.078,
        ult_stress=700.0,
    )

    assert N.shape == (3,)
    assert sigma_aeq.shape == (3,)
    # More severe loading should not increase life
    assert N[0] >= N[1] >= N[2]
    # sigma_aeq increases with mean stress under Goodman
    assert sigma_aeq[0] <= sigma_aeq[1] <= sigma_aeq[2]


def test_warns_and_clips_minimum_one_cycle() -> None:
    """If σ_a,eq > σ_f, warn and clip N to at least 1."""
    with pytest.warns(RuntimeWarning, match="exceeds σ_f"):
        N, sigma_aeq = goodman_fn(
            stress_amp=300.0,
            mean_stress=0.0,
            fat_strength_coef=200.0,  # small σ_f -> σ_a,eq > σ_f
            fat_strength_exp=-0.08,
            ult_stress=700.0,
        )
    assert int(N) >= 1
    assert float(sigma_aeq) > 200.0


def test_invalid_denominator_raises() -> None:
    """Denominator must be positive: (σ_m / σ_UTS) < 1."""
    with pytest.raises(ValueError, match="denominator became non-positive"):
        _ = goodman_fn(
            stress_amp=100.0,
            mean_stress=700.0,  # equals UTS -> denom = 0
            fat_strength_coef=400.0,
            fat_strength_exp=-0.1,
            ult_stress=700.0,
        )


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        _ = goodman_fn(
            stress_amp=0.0,  # must be > 0
            mean_stress=0.0,
            fat_strength_coef=400.0,
            fat_strength_exp=-0.1,
            ult_stress=700.0,
        )
    with pytest.raises(ValueError):
        _ = goodman_fn(
            stress_amp=100.0,
            mean_stress=0.0,
            fat_strength_coef=-1.0,  # must be > 0
            fat_strength_exp=-0.1,
            ult_stress=700.0,
        )
    with pytest.raises(ValueError):
        _ = goodman_fn(
            stress_amp=100.0,
            mean_stress=0.0,
            fat_strength_coef=400.0,
            fat_strength_exp=0.0,  # b must be < 0
            ult_stress=700.0,
        )
