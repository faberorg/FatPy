"""Goodman mean-stress correction for uniaxial S–N life.

This module exposes :func:`s_eqa_goodman`, which computes the Goodman
equivalent stress amplitude and the corresponding cycles to failure:

    sigma_aeq = sigma_a / (1 - sigma_m / sigma_UTS)
    N         = 0.5 * (sigma_aeq / sigma_f) ** (1 / b)

The implementation is vectorized (NumPy broadcasting) and unit-agnostic
(provided all inputs use consistent units). It returns `(N, sigma_aeq)`,
where `N` is rounded to the nearest integer and clipped to a minimum of 1.

"""

from __future__ import annotations

from typing import Union
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


Number = Union[int, float]


def s_eqa_goodman(
    stress_amp: ArrayLike,
    mean_stress: ArrayLike,
    fat_strength_coef: ArrayLike | Number,
    fat_strength_exp: ArrayLike | Number,
    ult_stress: ArrayLike | Number,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Stress-life (S–N) prediction using the Goodman mean-stress correction.

    Implements the Goodman mean-stress correction (Suresh, 1998):

        σ_a,eq = σ_a / (1 - σ_m / σ_UTS)
        N      = 0.5 * (σ_a,eq / σ_f) ** (1 / b)

    where `b < 0` is the Basquin exponent.

    Args:
        stress_amp: σ_a. Stress amplitude(s). Must be > 0.
        mean_stress: σ_m. Mean stress(es).
        fat_strength_coef: σ_f. Basquin fatigue strength coefficient(s). Must be > 0.
        fat_strength_exp: b. Basquin fatigue strength exponent(s). Must be < 0.
        ult_stress: σ_UTS. Ultimate tensile strength(s). Must be > 0.

    Returns:
        (N, sigma_aeq):
            N: integer ndarray of predicted cycles to failure (rounded to nearest int,
               clipped to a minimum of 1).
            sigma_aeq: float ndarray of equivalent stress amplitude(s) per Goodman.

    Raises:
        ValueError: If inputs violate constraints (e.g., b >= 0, σ_f <= 0, σ_UTS <= 0,
            σ_a <= 0) or if any (σ_m / σ_UTS) >= 1, which would make the Goodman
            denominator non-positive.

    Notes:
        * Vectorized: all arguments broadcast via NumPy rules.
        * If σ_a,eq > σ_f anywhere (which would give N < 1), a warning is issued.
          In that case, N is clipped to a minimum of 1.
        * Units must be consistent throughout (e.g., MPa or Pa).
    """
    # Convert to arrays (float64) for safe broadcasting / math
    sa = np.asarray(stress_amp, dtype=np.float64)
    sm = np.asarray(mean_stress, dtype=np.float64)
    sf = np.asarray(fat_strength_coef, dtype=np.float64)
    b = np.asarray(fat_strength_exp, dtype=np.float64)
    uts = np.asarray(ult_stress, dtype=np.float64)

    # Basic validation (constraints)
    if np.any(sa <= 0):
        raise ValueError("stress_amp (σ_a) must be > 0.")
    if np.any(sf <= 0):
        raise ValueError("fat_strength_coef (σ_f) must be > 0.")
    if np.any(uts <= 0):
        raise ValueError("ult_stress (σ_UTS) must be > 0.")
    if np.any(b >= 0):
        raise ValueError("fat_strength_exp (b) must be < 0 for Basquin.")

    # Goodman denominator
    denom = 1.0 - (sm / uts)
    if np.any(denom <= 0.0):
        raise ValueError(
            "Invalid mean stress: (σ_m / σ_UTS) must be < 1. "
            "Goodman denominator became non-positive."
        )

    # Equivalent stress amplitude
    sigma_aeq = sa / denom

    # Warn if σ_a,eq exceeds σ_f (implies N < 1)
    if np.any(sigma_aeq > sf):
        warnings.warn(
            "Equivalent stress amplitude exceeds σ_f; "
            "predicted life would be < 1 cycle. Clipping N to a minimum of 1.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Basquin inversion: N = 0.5 * (σ_a,eq / σ_f) ** (1 / b)
    with np.errstate(divide="raise", invalid="raise"):
        N_float = 0.5 * (sigma_aeq / sf) ** (1.0 / b)

    # Enforce a minimum of 1 cycle to avoid N < 1.0
    N_float = np.maximum(N_float, 1.0)

    # Integer cycles as in your original function
    N = np.rint(N_float).astype(np.int64)

    # Return both N and σ_a,eq
    return N, sigma_aeq
