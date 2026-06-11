"""Uniaxial fatigue criteria methods for the stress-life approach.

This module contains criteria for uniaxial high-cycle fatigue that incorporate
mean-stress effects via equivalent stress amplitudes. It adjusts the stress amplitude
using models such as Goodman, Gerber, and Soderberg to provide more accurate
fatigue-life predictions when mean stresses significantly affect material endurance.

For more information, you can refer to the following resource:

[PAPUGA, Jan, et al. Mean stress effect in stress-life fatigue prediction re-evaluated.
In: MATEC web of conferences. EDP Sciences, 2018. p. 10018.](https://doi.org/10.1051/matecconf/201816510018).
"""

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _asme_correction_method(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    yield_strength: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    """Calculate equivalent stress amplitude using ASME criterion."""
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    yield_strength_arr = np.asarray(yield_strength, dtype=np.float64)

    asme_eq_amp = (
        stress_amp_arr / (1 - (mean_stress_arr / yield_strength_arr) ** 2) ** 0.5
    )

    return asme_eq_amp


# Preset tolerances for all np.isclose() functions in mean stress correction methods
_RTOL = 0.001
_ATOL = 1e-5


def calc_stress_eq_amp_asme(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    yield_strength: ArrayLike | np.float64,
    allow_neg_mean_stress: bool = True,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using ASME criterion.

    ??? abstract "Math Equations"
        The ASME equivalent stress amplitude is calculated as:

        $$
        \sigma_{aeq}=\frac{\sigma_a}{\left[1-\left(\frac{\sigma_m}{R_e}\right)^2\right]^{1/2}}
        $$

    Args:
        stress_amp: The stress amplitude values.
            Leading dimensions are preserved.
        mean_stress: The mean stress values. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        yield_strength: The yield strength values. Must be broadcastable with
            stress_amp and mean_stress. Leading dimensions are preserved.
        allow_neg_mean_stress: A flag to control the calculation method.
            Defaults to True. If set to False, the equivalent stress amplitude will be
            set equal to the original stress amplitude for cases where the mean stress
            is negative, ignoring the correction.
        rtol: Relative tolerance for checking if mean stress magnitude is close to
            yield strength.
        atol: Absolute tolerance for checking if mean stress magnitude is close to
            yield strength.

    Raises:
        Warning: If stress amplitude is negative ($\sigma_a < 0$).
        ValueError: If yield strength is not positive ($R_e > 0$).
        ValueError: If mean stress magnitude exceeds yield strength,
            which would produce a negative value under the square root.
            ($|\sigma_m| > R_e$)
        ValueError: If mean stress magnitude is close to yield strength
            (within tolerance), the equivalent stress amplitude tends to infinity.
            ($\left|\frac{\sigma_m}{R_e}\right| \approx 1.0$ within tolerance).

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    yield_strength_arr = np.asarray(yield_strength, dtype=np.float64)

    if np.any(yield_strength_arr <= 0):
        raise ValueError("Yield strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = np.abs(mean_stress_arr) / yield_strength_arr

    if np.any(ratio > 1.0):
        raise ValueError(
            "Mean stress magnitude exceeds yield strength, which produces a negative"
            " value under the square root."
        )

    if np.any(np.isclose(ratio, 1.0, rtol=rtol, atol=atol)):
        raise ValueError(
            "Mean stress magnitude is close to yield strength, this results in "
            "infinite equivalent stress amplitude."
        )

    if np.any(stress_amp_arr < 0):
        warnings.warn(
            "Stress amplitude is negative.",
            UserWarning,
            stacklevel=2,
        )

    eq_stress_amp_arr = _asme_correction_method(
        stress_amp_arr, mean_stress_arr, yield_strength_arr
    )

    # If allow_neg_mean_stress is False, set equivalent stress amplitude = to original
    if not allow_neg_mean_stress:
        eq_stress_amp_arr = np.where(
            mean_stress_arr < 0, stress_amp_arr, eq_stress_amp_arr
        )

    return eq_stress_amp_arr


def _bagci_correction_method(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    yield_strength: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    """Calculate equivalent stress amplitude using Bagci criterion."""
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    yield_strength_arr = np.asarray(yield_strength, dtype=np.float64)

    bagci_eq_amp = stress_amp_arr / (1 - (mean_stress_arr / yield_strength_arr) ** 4)

    return bagci_eq_amp


def calc_stress_eq_amp_bagci(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    yield_strength: ArrayLike | np.float64,
    allow_neg_mean_stress: bool = True,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Bagci criterion.

    ??? abstract "Math Equations"
        The Bagci equivalent stress amplitude is calculated as:

        $$
        \sigma_{aeq}=\frac{\sigma_a}{1-\left(\frac{\sigma_m}{R_e}\right)^4}
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        yield_strength: Array-like of yield strengths. Must be broadcastable with
            stress_amp and mean_stress. Leading dimensions are preserved.
        allow_neg_mean_stress: A flag to control the calculation method.
            Defaults to True. If set to False, the equivalent stress amplitude will be
            set equal to the original stress amplitude for cases where the mean stress
            is negative, ignoring the correction.
        rtol: Relative tolerance for checking if mean stress magnitude is close to
            yield strength.
        atol: Absolute tolerance for checking if mean stress magnitude is close to
            yield strength.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress magnitude exceeds yield strength ($|\sigma_m| > R_e$).
        Warning: If stress amplitude is negative ($\sigma_a < 0$).
        ValueError: If yield strength is not positive ($R_e > 0$).
        ValueError: If mean stress magnitude is close to yield strength
            (within tolerance), the equivalent stress amplitude tends to infinity.
            ($\left|\frac{\sigma_m}{R_e}\right| \approx 1.0$ within tolerance).
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    yield_strength_arr = np.asarray(yield_strength, dtype=np.float64)

    if np.any(yield_strength_arr <= 0):
        raise ValueError("Yield strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = np.abs(mean_stress_arr) / yield_strength_arr

    if np.any(np.isclose(ratio, 1.0, rtol=rtol, atol=atol)):
        raise ValueError(
            "Mean stress magnitude is close to yield strength, this results in "
            "infinite equivalent stress amplitude."
        )

    if np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress magnitude exceeds yield strength.",
            UserWarning,
            stacklevel=2,
        )

    if np.any(stress_amp_arr < 0):
        warnings.warn(
            "Stress amplitude is negative.",
            UserWarning,
            stacklevel=2,
        )

    eq_stress_amp_arr = _bagci_correction_method(
        stress_amp_arr, mean_stress_arr, yield_strength_arr
    )

    # If allow_neg_mean_stress is False, set equivalent stress amplitude = to original
    if not allow_neg_mean_stress:
        eq_stress_amp_arr = np.where(
            mean_stress_arr < 0, stress_amp_arr, eq_stress_amp_arr
        )

    return eq_stress_amp_arr


def _gerber_correction_method(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    ult_tensile_strength: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    """Calculate equivalent stress amplitude using Gerber criterion."""
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    ult_tensile_strength_arr = np.asarray(ult_tensile_strength, dtype=np.float64)

    gerber_eq_amp = stress_amp_arr / (
        1 - (mean_stress_arr / ult_tensile_strength_arr) ** 2
    )

    return gerber_eq_amp


def calc_stress_eq_amp_gerber(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    ult_tensile_strength: ArrayLike | np.float64,
    allow_neg_mean_stress: bool = True,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Gerber criterion.

    ??? abstract "Math Equations"
        The Gerber equivalent stress amplitude is calculated as:

        $$
        \displaystyle\sigma_{aeq}=\frac{\sigma_a}{1-\left(\frac{\sigma_m}{\sigma_{UTS}}
            \right)^2 }
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        ult_tensile_strength: Array-like of ultimate tensile strengths. Must be
            broadcastable with stress_amp and mean_stress.
            Leading dimensions are preserved.
        allow_neg_mean_stress: A flag to control the calculation method.
            Defaults to True. If set to False, the equivalent stress amplitude will be
            set equal to the original stress amplitude for cases where the mean stress
            is negative, ignoring the correction.
        rtol: Relative tolerance for checking if mean stress magnitude is close to
            ultimate tensile strength.
        atol: Absolute tolerance for checking if mean stress magnitude is close to
            ultimate tensile strength.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress magnitude exceeds ultimate tensile strength
            ($|\sigma_m| > \sigma_{UTS}$).
        Warning: If stress amplitude is negative ($\sigma_a < 0$).
        ValueError: If ultimate tensile strength is not positive ($\sigma_{UTS} > 0$).
        ValueError: If mean stress magnitude is close to ultimate tensile strength
            (within tolerance), the equivalent stress amplitude tends to infinity.
            ($\left|\frac{\sigma_m}{\sigma_{UTS}}\right| \approx 1.0$ within tolerance).

    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    ult_tensile_strength_arr = np.asarray(ult_tensile_strength, dtype=np.float64)

    if np.any(ult_tensile_strength_arr <= 0):
        raise ValueError("Ultimate tensile strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = np.abs(mean_stress_arr) / ult_tensile_strength_arr

    if np.any(np.isclose(ratio, 1.0, rtol=rtol, atol=atol)):
        raise ValueError(
            "Mean stress magnitude is close to ultimate tensile strength, "
            "this results in infinite equivalent stress amplitude."
        )

    if np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress magnitude exceeds ultimate tensile strength. ",
            UserWarning,
            stacklevel=2,
        )

    if np.any(stress_amp_arr < 0):
        warnings.warn(
            "Stress amplitude is negative.",
            UserWarning,
            stacklevel=2,
        )

    eq_stress_amp_arr = _gerber_correction_method(
        stress_amp_arr, mean_stress_arr, ult_tensile_strength_arr
    )

    # If allow_neg_mean_stress is False, set equivalent stress amplitude = to original
    if not allow_neg_mean_stress:
        eq_stress_amp_arr = np.where(
            mean_stress_arr < 0, stress_amp_arr, eq_stress_amp_arr
        )

    return eq_stress_amp_arr


def _linear_correction_method(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    material_parameter: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    """Calculate equivalent stress amplitude using linear correction."""
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    material_parameter_arr = np.asarray(material_parameter, dtype=np.float64)

    linear_eq_amp = stress_amp_arr / (1 - (mean_stress_arr / material_parameter_arr))

    return linear_eq_amp


def calc_stress_eq_amp_goodman(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    ult_tensile_strength: ArrayLike | np.float64,
    allow_neg_mean_stress: bool = True,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Goodman criterion.

    ??? abstract "Math Equations"
        The Goodman equivalent stress amplitude is calculated as:

        $$
        \displaystyle\sigma_{aeq}=\frac{\sigma_a}{1-\frac{\sigma_m}{\sigma_{UTS}}}
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        ult_tensile_strength: Array-like of ultimate tensile strengths. Must be
            broadcastable with stress_amp and mean_stress.
            Leading dimensions are preserved.
        allow_neg_mean_stress: A flag to control the calculation method.
            Defaults to True. If set to False, the equivalent stress amplitude will be
            set equal to the original stress amplitude for cases where the mean stress
            is negative, ignoring the correction.
        rtol: Relative tolerance for checking if mean stress magnitude is close to
            ultimate tensile strength.
        atol: Absolute tolerance for checking if mean stress magnitude is close to
            ultimate tensile strength.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress magnitude exceeds ultimate tensile strength
            ($|\sigma_m| > \sigma_{UTS}$).
        Warning: If stress amplitude is negative ($\sigma_a < 0$).
        ValueError: If ultimate tensile strength is not positive ($\sigma_{UTS} > 0$).
        ValueError: If mean stress magnitude is close to ultimate tensile strength
            (within tolerance), the equivalent stress amplitude tends to infinity.
            ($\left|\frac{\sigma_m}{\sigma_{UTS}}\right| \approx 1.0$ within tolerance).
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    ult_tensile_strength_arr = np.asarray(ult_tensile_strength, dtype=np.float64)

    if np.any(ult_tensile_strength_arr <= 0):
        raise ValueError("Ultimate tensile strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = abs(mean_stress_arr) / ult_tensile_strength_arr

    if np.any(np.isclose(ratio, 1.0, rtol=rtol, atol=atol)):
        raise ValueError(
            "Mean stress magnitude is close to ultimate tensile strength, "
            "this results in infinite equivalent stress amplitude."
        )

    if np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress magnitude exceeds ultimate tensile strength. ",
            UserWarning,
            stacklevel=2,
        )

    if np.any(stress_amp_arr < 0):
        warnings.warn(
            "Stress amplitude is negative.",
            UserWarning,
            stacklevel=2,
        )

    eq_stress_amp_arr = _linear_correction_method(
        stress_amp_arr, mean_stress_arr, ult_tensile_strength_arr
    )

    # If allow_neg_mean_stress is False, set equivalent stress amplitude = to original
    if not allow_neg_mean_stress:
        eq_stress_amp_arr = np.where(
            mean_stress_arr < 0, stress_amp_arr, eq_stress_amp_arr
        )

    return eq_stress_amp_arr


def calc_stress_eq_amp_half_slope(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    ult_tensile_strength: ArrayLike | np.float64,
    allow_neg_mean_stress: bool = True,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using a half-slope mean stress correction.

    ??? abstract "Math Equations"
        The half-slope corrected equivalent stress amplitude is calculated as:

        $$
        \sigma_{aeq}=\frac{\sigma_a}{1 - \frac{\sigma_m}{2 \cdot \sigma_{UTS}}}
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        ult_tensile_strength: Array-like of ultimate tensile strengths. Must be
            broadcastable with stress_amp and mean_stress. Leading dimensions are
            preserved.
        allow_neg_mean_stress: A flag to control the calculation method.
            Defaults to True. If set to False, the equivalent stress amplitude will be
            set equal to the original stress amplitude for cases where the mean stress
            is negative, ignoring the correction.
        rtol: Relative tolerance for checking if mean stress magnitude is close to
            ultimate tensile strength.
        atol: Absolute tolerance for checking if mean stress magnitude is close to
            ultimate tensile strength.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress magnitude exceeds the ultimate tensile strength
            ($|\sigma_m| > \sigma_{UTS}$).
        Warning: If stress amplitude is negative ($\sigma_a < 0$).
        ValueError: If ultimate tensile strength is not positive ($\sigma_{UTS} > 0$).
        ValueError: If mean stress magnitude is close to double of the ultimate tensile
            strength, (within tolerance), the equivalent stress amplitude tends to
            infinity. ($\left|\frac{\sigma_m}{\sigma_{UTS}}\right| \approx 2.0$ within
            tolerance).

    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    ult_tensile_strength_arr = np.asarray(ult_tensile_strength, dtype=np.float64)

    if np.any(ult_tensile_strength_arr <= 0):
        raise ValueError("Ultimate tensile strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = np.abs(mean_stress_arr) / ult_tensile_strength_arr

    if np.any(np.isclose(ratio, 2.0, rtol=rtol, atol=atol)):
        raise ValueError(
            "Mean stress magnitude is close to double of the ultimate tensile strength,"
            " this results in infinite equivalent stress amplitude."
        )

    if np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress magnitude exceeds the ultimate tensile strength. ",
            UserWarning,
            stacklevel=2,
        )

    if np.any(stress_amp_arr < 0):
        warnings.warn(
            "Stress amplitude is negative.",
            UserWarning,
            stacklevel=2,
        )

    eq_stress_amp_arr = _linear_correction_method(
        stress_amp_arr, mean_stress_arr, 2 * ult_tensile_strength_arr
    )

    # If allow_neg_mean_stress is False, set equivalent stress amplitude = to original
    if not allow_neg_mean_stress:
        eq_stress_amp_arr = np.where(
            mean_stress_arr < 0, stress_amp_arr, eq_stress_amp_arr
        )

    return eq_stress_amp_arr


def calc_stress_eq_amp_linear(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    stress_param_m: ArrayLike | np.float64,
    allow_neg_mean_stress: bool = True,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using a linear mean stress correction.

    ??? abstract "Math Equations"
        The linearly corrected equivalent stress amplitude is calculated as:

        $$
        \sigma_{aeq}=\frac{\sigma_a}{1 - \frac{\sigma_m}{M}}
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        stress_param_m: Array-like of material stress parameters M.
            Must be broadcastable with stress_amp and mean_stress.
            Leading dimensions are preserved.
        allow_neg_mean_stress: A flag to control the calculation method.
            Defaults to True. If set to False, the equivalent stress amplitude will be
            set equal to the original stress amplitude for cases where the mean stress
            is negative, ignoring the correction.
        rtol: Relative tolerance for checking if mean stress magnitude is close to
            stress parameter M.
        atol: Absolute tolerance for checking if mean stress magnitude is close to
            stress parameter M.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress magnitude exceeds material stress parameter M
            ($|\sigma_m| > M$).
        Warning: If stress amplitude is negative ($\sigma_a < 0$).
        ValueError: If material stress parameter M is not positive ($M > 0$).
        ValueError: If mean stress magnitude is close to stress parameter M
            (within tolerance), the equivalent stress amplitude tends to infinity.
            ($\left|\frac{\sigma_m}{M}\right| \approx 1.0$ within tolerance).
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    stress_param_m_arr = np.asarray(stress_param_m, dtype=np.float64)

    if np.any(stress_param_m_arr <= 0):
        raise ValueError("Material stress parameter M must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = abs(mean_stress_arr) / stress_param_m_arr

    if np.any(np.isclose(ratio, 1.0, rtol=rtol, atol=atol)):
        raise ValueError(
            "Mean stress magnitude is close to stress parameter M, "
            "this results in infinite equivalent stress amplitude."
        )

    if np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress magnitude exceeds material stress parameter M. ",
            UserWarning,
            stacklevel=2,
        )

    if np.any(stress_amp_arr < 0):
        warnings.warn(
            "Stress amplitude is negative.",
            UserWarning,
            stacklevel=2,
        )

    eq_stress_amp_arr = _linear_correction_method(
        stress_amp_arr, mean_stress_arr, stress_param_m_arr
    )

    # If allow_neg_mean_stress is False, set equivalent stress amplitude = to original
    if not allow_neg_mean_stress:
        eq_stress_amp_arr = np.where(
            mean_stress_arr < 0, stress_amp_arr, eq_stress_amp_arr
        )

    return eq_stress_amp_arr


def calc_stress_eq_amp_morrow(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    fat_strength_coef: ArrayLike | np.float64,
    allow_neg_mean_stress: bool = True,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Morrow criterion.

    ??? abstract "Math Equations"
        The Morrow equivalent stress amplitude is calculated as:

        $$
        \displaystyle\sigma_{aeq}=\frac{\sigma_a}{1-\frac{\sigma_m}{\sigma_{true}} }
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        fat_strength_coef: Array-like of fatigue strength coefficients. Must be
            broadcastable with stress_amp and mean_stress. Leading dimensions
            are preserved.
        allow_neg_mean_stress: A flag to control the calculation method.
            Defaults to True. If set to False, the equivalent stress amplitude will be
            set equal to the original stress amplitude for cases where the mean stress
            is negative, ignoring the correction
        rtol: Relative tolerance for checking if mean stress magnitude is close to
            fatigue strength coefficient.
        atol: Absolute tolerance for checking if mean stress magnitude is close to
            fatigue strength coefficient.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress magnitude exceeds fatigue strength coefficient
            ($|\sigma_m| > \sigma_{f}'$).
        Warning: If stress amplitude is negative ($\sigma_a < 0$).
        ValueError: If fatigue strength coefficient is not positive ($\sigma_{f}' > 0$).
        ValueError: If mean stress magnitude is close to fatigue strength coefficient
            (within tolerance), the equivalent stress amplitude tends to infinity.
            ($\left|\frac{\sigma_m}{\sigma_{f}'}\right| \approx 1.0$ within tolerance).
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    fat_strength_coef_arr = np.asarray(fat_strength_coef, dtype=np.float64)

    if np.any(fat_strength_coef_arr <= 0):
        raise ValueError("Fatigue strength coefficient must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = np.abs(mean_stress_arr) / fat_strength_coef_arr

    if np.any(np.isclose(ratio, 1.0, rtol=rtol, atol=atol)):
        raise ValueError(
            "Mean stress magnitude is close to fatigue strength coefficient, "
            "this results in infinite equivalent stress amplitude."
        )

    if np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress magnitude exceeds fatigue strength coefficient. ",
            UserWarning,
            stacklevel=2,
        )

    if np.any(stress_amp_arr < 0):
        warnings.warn(
            "Stress amplitude is negative.",
            UserWarning,
            stacklevel=2,
        )

    eq_stress_amp_arr = _linear_correction_method(
        stress_amp_arr, mean_stress_arr, fat_strength_coef_arr
    )

    # If allow_neg_mean_stress is False, set equivalent stress amplitude = to original
    if not allow_neg_mean_stress:
        eq_stress_amp_arr = np.where(
            mean_stress_arr < 0, stress_amp_arr, eq_stress_amp_arr
        )

    return eq_stress_amp_arr


def calc_stress_eq_amp_soderberg(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    yield_strength: ArrayLike | np.float64,
    allow_neg_mean_stress: bool = True,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Soderberg criterion.

    ??? abstract "Math Equations"
        The Soderberg equivalent stress amplitude is calculated as:

        $$
        \sigma_{aeq}=\frac{\sigma_a}{1-\frac{\sigma_m}{R_e}}
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        yield_strength: Array-like of yield strengths. Must be broadcastable with
            stress_amp and mean_stress. Leading dimensions are preserved.
        allow_neg_mean_stress: A flag to control the calculation method.
            Defaults to True. If set to False, the equivalent stress amplitude will be
            set equal to the original stress amplitude for cases where the mean stress
            is negative, ignoring the correction.
        rtol: Relative tolerance for checking if mean stress magnitude is close to
            yield strength.
        atol: Absolute tolerance for checking if mean stress magnitude is close to
            yield strength.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress magnitude exceeds yield strength ($|\sigma_m| > R_e$).
        Warning: If stress amplitude is negative ($\sigma_a < 0$).
        ValueError: If yield strength is not positive ($R_e > 0$).
        ValueError: If mean stress magnitude is close to yield strength
            (within tolerance), the equivalent stress amplitude tends to infinity.
            ($\left|\frac{\sigma_m}{R_e}\right| \approx 1.0$ within tolerance).
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    yield_strength_arr = np.asarray(yield_strength, dtype=np.float64)

    if np.any(yield_strength_arr <= 0):
        raise ValueError("Yield strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = np.abs(mean_stress_arr) / yield_strength_arr

    if np.any(np.isclose(ratio, 1.0, rtol=rtol, atol=atol)):
        raise ValueError(
            "Mean stress magnitude is close to yield strength, this results in "
            "infinite equivalent stress amplitude."
        )

    if np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress magnitude exceeds yield strength. ",
            UserWarning,
            stacklevel=2,
        )

    if np.any(stress_amp_arr < 0):
        warnings.warn(
            "Stress amplitude is negative.",
            UserWarning,
            stacklevel=2,
        )

    eq_stress_amp_arr = _linear_correction_method(
        stress_amp_arr, mean_stress_arr, yield_strength_arr
    )

    # If allow_neg_mean_stress is False, set equivalent stress amplitude = to original
    if not allow_neg_mean_stress:
        eq_stress_amp_arr = np.where(
            mean_stress_arr < 0, stress_amp_arr, eq_stress_amp_arr
        )

    return eq_stress_amp_arr


def _smith_correction_method(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    ult_tensile_strength: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    """Calculate equivalent stress amplitude using Smith criterion."""
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    ult_tensile_strength_arr = np.asarray(ult_tensile_strength, dtype=np.float64)

    smith_eq_amp = (
        stress_amp_arr * (1 + mean_stress_arr / ult_tensile_strength_arr)
    ) / (1 - mean_stress_arr / ult_tensile_strength_arr)

    return smith_eq_amp


def calc_stress_eq_amp_smith(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    ult_tensile_strength: ArrayLike | np.float64,
    allow_neg_mean_stress: bool = True,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Smith criterion.

    ??? abstract "Math Equations"
        The Smith equivalent stress amplitude is calculated as:

        $$
        \sigma_{aeq}=\frac{\sigma_a \cdot \left(1 + \frac{\sigma_m}{\sigma_{UTS}}
        \right)}{1-\left(\frac{\sigma_m}{\sigma_{UTS}}\right)}
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        ult_tensile_strength: Array-like of ultimate tensile strengths. Must be
            broadcastable with stress_amp and mean_stress.
            Leading dimensions are preserved.
        allow_neg_mean_stress: A flag to control the calculation method.
            Defaults to True. If set to False, the equivalent stress amplitude will be
            set equal to the original stress amplitude for cases where the mean stress
            is negative, ignoring the correction.
        rtol: Relative tolerance for checking if mean stress magnitude is close to
            ultimate tensile strength.
        atol: Absolute tolerance for checking if mean stress magnitude is close to
            ultimate tensile strength.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress magnitude exceeds ultimate tensile strength
            ($\sigma_m > \sigma_{UTS}$).
        Warning: If stress amplitude is negative ($\sigma_a < 0$).
        ValueError: If ultimate tensile strength is not positive ($\sigma_{UTS} > 0$).
        ValueError: If mean stress magnitude is close to ultimate tensile strength
            (within tolerance), the equivalent stress amplitude tends to infinity.
            ($\left|\frac{\sigma_m}{\sigma_{UTS}}\right| \approx 1.0$ within tolerance).

    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    ult_tensile_strength_arr = np.asarray(ult_tensile_strength, dtype=np.float64)

    if np.any(ult_tensile_strength_arr <= 0):
        raise ValueError("Ultimate tensile strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = np.abs(mean_stress_arr / ult_tensile_strength_arr)

    if np.any(np.isclose(ratio, 1.0, rtol=rtol, atol=atol)):
        raise ValueError(
            "Mean stress magnitude is close to ultimate tensile strength, "
            "this results in infinite equivalent stress amplitude."
        )

    if np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress magnitude exceeds ultimate tensile strength. ",
            UserWarning,
            stacklevel=2,
        )

    if np.any(stress_amp_arr < 0):
        warnings.warn(
            "Stress amplitude is negative.",
            UserWarning,
            stacklevel=2,
        )

    eq_stress_amp_arr = _smith_correction_method(
        stress_amp_arr, mean_stress_arr, ult_tensile_strength_arr
    )

    # If allow_neg_mean_stress is False, set equivalent stress amplitude = to original
    if not allow_neg_mean_stress:
        eq_stress_amp_arr = np.where(
            mean_stress_arr < 0, stress_amp_arr, eq_stress_amp_arr
        )

    return eq_stress_amp_arr


def _swt_correction_method(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    """Calculate equivalent stress amplitude using Smith-Watson-Topper criterion."""
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)

    swt_eq_amp = np.sqrt(stress_amp_arr * (mean_stress_arr + stress_amp_arr))

    return swt_eq_amp


def calc_stress_eq_amp_swt(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    allow_neg_mean_stress: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Smith-Watson-Topper parameter.

    ??? abstract "Math Equations"
        The SWT equivalent stress amplitude is calculated as:

        $$
        \sigma_{aeq} = \sqrt{\sigma_{a} \cdot (\sigma_{m} + \sigma_{a})} \\
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        allow_neg_mean_stress: A flag to control the calculation method.
            Defaults to True. If set to False, the equivalent stress amplitude will be
            set equal to the original stress amplitude for cases where the mean stress
            is negative, ignoring the correction.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        ValueError: If stress amplitude is negative ($\sigma_a < 0$).
        ValueError: If the validity condition $\sigma_a + \sigma_m >0$ is not satisfied.

    ??? note "Validity Condition"
        The SWT parameter is valid when $\sigma_a + \sigma_m > 0$, ensuring that the
        maximum stress in the cycle is positive (tensile). When this condition is
        not met, a ValueError is raised.

    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)

    # Check for negative stress amplitudes
    if np.any(stress_amp_arr < 0):
        raise ValueError("Stress amplitude must be non-negative")

    # Check validity condition: σₐ + σₘ > 0
    invalid_condition = stress_amp_arr + mean_stress_arr <= 0

    if np.any(invalid_condition):
        raise ValueError(
            r"Smith-Watson-Topper parameter validity condition $\sigma_a + \sigma_m >0$"
            " not satisfied for some data points. The SWT approach may not be "
            "appropriate for compressive-dominated loading conditions."
        )

    eq_stress_amp_arr = _swt_correction_method(stress_amp_arr, mean_stress_arr)

    # If allow_neg_mean_stress is False, set equivalent stress amplitude = to original
    if not allow_neg_mean_stress:
        eq_stress_amp_arr = np.where(
            mean_stress_arr < 0, stress_amp_arr, eq_stress_amp_arr
        )

    return eq_stress_amp_arr


def _walker_correction_method(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    walker_param: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    """Calculate equivalent stress amplitude using Walker criterion."""
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    walker_param_arr = np.asarray(walker_param, dtype=np.float64)

    walker_eq_amp = (stress_amp_arr + mean_stress_arr) ** (
        1 - walker_param_arr
    ) * stress_amp_arr**walker_param_arr

    return walker_eq_amp


def calc_stress_eq_amp_walker(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    walker_param: ArrayLike | np.float64,
    allow_neg_mean_stress: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Walker criterion.

    ??? abstract "Math Equations"
        The Walker equivalent stress amplitude is calculated as:

        $$
        \displaystyle\sigma_{aeq}=\left(\sigma_a+\sigma_m\right)^{1-\gamma} \cdot
            \sigma_a^{\gamma}
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        walker_param: Array-like of Walker exponents ($\gamma$). Must be broadcastable
            with stress_amp and mean_stress. Leading dimensions are preserved.
        allow_neg_mean_stress: A flag to control the calculation method.
            Defaults to True. If set to False, the equivalent stress amplitude will be
            set equal to the original stress amplitude for cases where the mean stress
            is negative, ignoring the correction.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        ValueError: If stress amplitude is negative ($\sigma_a < 0$).
        ValueError: If the validity condition $\sigma_a + \sigma_m >0$ is not satisfied.
        ValueError: When the condition $\gamma$ in [0, 1] is not satisfied.

    ??? note "Validity Condition"
        The Walker method is valid when $\sigma_a + \sigma_m > 0$, ensuring that the
        maximum stress in the cycle is positive (tensile). When this condition is
        not met, a ValueError is raised.
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    walker_param_arr = np.asarray(walker_param, dtype=np.float64)

    # Check for negative stress amplitudes
    if np.any(stress_amp_arr < 0):
        raise ValueError("Stress amplitude must be non-negative")

    # Check validity condition: σₐ + σₘ > 0
    invalid_condition = stress_amp_arr + mean_stress_arr <= 0

    if np.any(invalid_condition):
        raise ValueError(
            r"Walker method validity condition $\sigma_a + \sigma_m >0$ not "
            "satisfied for some data points. The Walker approach may not be "
            "appropriate for compressive-dominated loading conditions."
        )

    # Check validity of Walker parameter: γ' in range [0, 1]
    invalid_condition = (walker_param_arr < 0) | (walker_param_arr > 1)
    if np.any(invalid_condition):
        raise ValueError(r"Walker parameter ($\gamma$) must be in the range [0, 1]. ")

    eq_stress_amp_arr = _walker_correction_method(
        stress_amp_arr, mean_stress_arr, walker_param_arr
    )

    # If allow_neg_mean_stress is False, set equivalent stress amplitude = to original
    if not allow_neg_mean_stress:
        eq_stress_amp_arr = np.where(
            mean_stress_arr < 0, stress_amp_arr, eq_stress_amp_arr
        )

    return eq_stress_amp_arr
