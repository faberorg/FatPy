"""Uniaxial fatigue criteria methods for the stress-life approach.

Contains criteria that address uniaxial high-cycle fatigue by incorporating the mean
stress effect through an equivalent stress amplitude approach. By adjusting the stress
amplitude to account for mean stress influences—using models such as Goodman, Gerber,
or Soderberg—they enable more accurate fatigue life predictions where mean stresses
significantly affect material endurance.

For more information you can refer to the following resource:
https://doi.org/10.1051/matecconf/201816510018
"""

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

# TODO: wrapper functionality with allowing negative mean stresses


def _asme_correction_method(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    yield_strength: ArrayLike | np.float64,  # TODO? Array or float?
) -> NDArray[np.float64]:
    """Calculate equivalent stress amplitude using ASME criterion."""
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    yield_strength_arr = np.asarray(yield_strength, dtype=np.float64)

    asme_eq_amp = (
        stress_amp_arr / (1 - (mean_stress_arr / yield_strength_arr) ** 2) ** 0.5
    )

    return asme_eq_amp


def calc_stress_eq_amp_asme(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    yield_strength: ArrayLike | np.float64,
    allow_neg_mean_stress: bool = True,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using ASME criterion.

    ??? abstract "Math Equations"
        The ASME equivalent stress amplitude is calculated as:

        $$
        \sigma_{aeq}=\frac{\sigma_a}{\left[1-\left(\frac{\sigma_m}{R_e}\right)^2\right]^{1/2}}
        $$

    Args:
        stress_amp(ArrayLike): The stress amplitude values.
            Leading dimensions are preserved.
        mean_stress(ArrayLike): The mean stress values. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        yield_strength(ArrayLike): The yield strength values. Must be broadcastable with
            stress_amp and mean_stress. Leading dimensions are preserved.
        allow_neg_mean_stress(bool, optional): A flag to control the calculation method.
            Defaults to True. If set to False, the equivalent stress amplitude will be
            set equal to the original stress amplitude for cases where the mean stress
            is negative, ignoring the correction.

    Raises:
        ValueError: If yield strength is not positive.
        ValueError: If mean stress magnitude is equal or greater to yield strength,
            resulting in infinite equivalent stress amplitude.

    Returns:
        NDArray[np.float64]: The calculated equivalent amplitude stress values.
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    yield_strength_arr = np.asarray(yield_strength, dtype=np.float64)

    if np.any(yield_strength_arr <= 0):
        raise ValueError("Yield strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = np.abs(mean_stress_arr) / yield_strength_arr
    if np.any(ratio >= 1.0):
        raise ValueError("Mean stress magnitude equal or greater than yield strength.")

    eq_stress_amp_arr = _asme_correction_method(
        stress_amp_arr, mean_stress_arr, yield_strength_arr
    )

    if not allow_neg_mean_stress:
        eq_stress_amp_arr = np.where(
            mean_stress_arr < 0, stress_amp_arr, eq_stress_amp_arr
        )

    return eq_stress_amp_arr


def calc_stress_eq_amp_ASME(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    yield_strength: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using ASME criterion.

    ??? info "ASME Use-case"
        The ASME criterion accounts for mean stress effects in high-cycle fatigue
        by modifying the stress amplitude based on the yield strength using a
        quadratic, square-root denominator relationship.

    ??? abstract "Math Equations"
        The ASME equivalent stress amplitude is calculated as:

        $$
        \sigma_{aeq}=\frac{\sigma_a}{\left[1-\left(\frac{\sigma_m}{R_e}\right)^2\right]^{1/2}}
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        yield_strength: Array-like of yield strengths. Must be broadcastable with
            stress_amp and mean_stress. Leading dimensions are preserved.

    Raises:
        ValueError: If yield strength is not positive.
        ValueError: If mean stress magnitude is equal or greater to yield strength,
            resulting in infinite equivalent stress amplitude.
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    yield_strength_arr = np.asarray(yield_strength, dtype=np.float64)

    if np.any(yield_strength_arr <= 0):
        raise ValueError("Yield strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = np.abs(mean_stress_arr) / yield_strength_arr
    if np.any(ratio >= 1.0):
        raise ValueError("Mean stress magnitude equal or greater than yield strength.")

    return stress_amp_arr / (1 - (mean_stress_arr / yield_strength_arr) ** 2) ** 0.5


def calc_stress_eq_amp_bagci(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    yield_strength: ArrayLike | np.float64,
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

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress magnitude exceeds yield strength.
        ValueError: If yield strength is not positive.
        ValueError: If mean stress magnitude is equal to yield strength,
            resulting in infinite equivalent stress amplitude.
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    yield_strength_arr = np.asarray(yield_strength, dtype=np.float64)

    if np.any(yield_strength_arr <= 0):
        raise ValueError("Yield strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = np.abs(mean_stress_arr) / yield_strength_arr
    if np.any(ratio == 1.0):
        raise ValueError(
            "Mean stress magnitude equals yield strength this would result in "
            "infinite equivalent stress amplitude."
        )
    elif np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress magnitude exceeds yield strength.",
            UserWarning,
            stacklevel=2,
        )

    return stress_amp_arr / (1 - (mean_stress_arr / yield_strength_arr) ** 4)


def calc_stress_eq_amp_gerber(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    ult_tensile_strength: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Gerber criterion.

    ??? info "Gerber Use-case"
        The Gerber criterion accounts for mean stress effects in high-cycle fatigue
        by modifying the stress amplitude based on the ultimate tensile strength.

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
        ult_tensile_strength: Array-like of ultimate tensile strengths. Must be broadcastable
            with stress_amp and mean_stress. Leading dimensions are preserved.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress magnitude exceeds ultimate tensile strength.
        ValueError: If ultimate tensile strength is not positive.
        ValueError: If mean stress magnitude is equal to ultimate tensile strength,
            resulting in infinite equivalent stress amplitude.

    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    ult_tensile_strength_arr = np.asarray(ult_tensile_strength, dtype=np.float64)

    if np.any(ult_tensile_strength_arr <= 0):
        raise ValueError("Ultimate tensile strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = np.abs(mean_stress_arr) / ult_tensile_strength_arr

    if np.any(ratio == 1.0):
        raise ValueError(
            "Mean stress magnitude equals ultimate tensile strength this would "
            "result in infinite equivalent stress amplitude."
        )
    elif np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress magnitude exceeds ultimate tensile strength. ",
            UserWarning,
            stacklevel=2,
        )

    return stress_amp_arr / (1 - (mean_stress_arr / ult_tensile_strength_arr) ** 2)


def calc_stress_eq_amp_goodman(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    ult_tensile_strength: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Goodman criterion.

    ??? info "Goodman Use-case"
        The Goodman criterion accounts for mean stress effects in high-cycle fatigue
        by modifying the stress amplitude based on the ultimate tensile strength using
        a linear relationship.

    ??? abstract "Math Equations"
        The Goodman equivalent stress amplitude is calculated as:

        $$
        \displaystyle\sigma_{aeq}=\frac{\sigma_a}{1-\frac{\sigma_m}{\sigma_{UTS}}}
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        ult_tensile_strength: Array-like of ultimate tensile strengths. Must be broadcastable
            with stress_amp and mean_stress. Leading dimensions are preserved.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress exceeds ultimate tensile strength.
        ValueError: If ultimate tensile strength is not positive.
        ValueError: If mean stress is equal to ultimate tensile strength, resulting in
            infinite equivalent stress amplitude.
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    ult_tensile_strength_arr = np.asarray(ult_tensile_strength, dtype=np.float64)

    if np.any(ult_tensile_strength_arr <= 0):
        raise ValueError("Ultimate tensile strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = mean_stress_arr / ult_tensile_strength_arr

    if np.any(ratio == 1.0):
        raise ValueError(
            "Mean stress equals ultimate tensile strength this would result in "
            "infinite equivalent stress amplitude."
        )
    elif np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress magnitude exceeds ultimate tensile strength. ",
            UserWarning,
            stacklevel=2,
        )

    return stress_amp_arr / (1 - mean_stress_arr / ult_tensile_strength_arr)


def calc_stress_eq_amp_half_slope(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    ult_tensile_strength: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using a half-slope mean stress correction.

    ??? info "Half-slope Use-case"
        A half-slope mean stress correction can be applied to account for mean
        stress effects in high-cycle fatigue by modifying the stress amplitude based
        on the ultimate tensile strength.

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

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress exceeds double of the ultimate tensile strength.
        ValueError: If ultimate tensile strength is not positive.
        ValueError: If mean stress is equal to double of the ultimate tensile strength,
            resulting in zero equivalent stress amplitude.

    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    ult_tensile_strength_arr = np.asarray(ult_tensile_strength, dtype=np.float64)

    if np.any(ult_tensile_strength_arr <= 0):
        raise ValueError("Ultimate tensile strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = mean_stress_arr / (2 * ult_tensile_strength_arr)
    if np.any(ratio == 1.0):
        raise ValueError(
            "Mean stress equals half of the ultimate tensile strength this would result"
            "in zero equivalent stress amplitude."
        )
    elif np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress exceeds half of the ultimate tensile strength. ",
            UserWarning,
            stacklevel=2,
        )
    return stress_amp_arr / (1 - mean_stress_arr / (2 * ult_tensile_strength_arr))


def calc_stress_eq_amp_linear(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    stress_param_M: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using a linear mean stress correction.

    ??? info "Linear Use-case"
        A simple linear mean stress correction can be applied to account for mean
        stress effects in high-cycle fatigue by modifying the stress amplitude based
        on the ultimate tensile strength.

    ??? abstract "Math Equations"
        The linearly corrected equivalent stress amplitude is calculated as:

        $$
        \sigma_{aeq}=\frac{\sigma_a}{1 - \frac{\sigma_m}{M}}
        $$
    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        stress_param_M: Array-like of material stress parameters M.
            Must be broadcastable with stress_amp and mean_stress.
            Leading dimensions are preserved.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress exceeds material stress parameter M.
        ValueError: If material stress parameter M is not positive.
        ValueError: If mean stress is equal to material stress parameter M, resulting in
            zero equivalent stress amplitude.

    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    stress_param_M_arr = np.asarray(stress_param_M, dtype=np.float64)

    if np.any(stress_param_M_arr <= 0):
        raise ValueError("Material stress parameter M must be positive")
    # Check if mean stress approaches or exceeds material parameter
    ratio = mean_stress_arr / stress_param_M_arr

    if np.any(ratio == 1.0):
        raise ValueError(
            "Mean stress equals material stress parameter M this would result in "
            "zero equivalent stress amplitude."
        )
    elif np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress exceeds material stress parameter M. ",
            UserWarning,
            stacklevel=2,
        )

    return stress_amp_arr / (1 - mean_stress_arr / stress_param_M_arr)


# todo! Check the name of the material parameter,issue calls it a true fracture stress but the paper calls it a fatigue strength coeficient, exel calls it a fat_strength_coef
def calc_stress_eq_amp_morrow(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    true_frac_stress: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Morrow criterion.

    ??? info "Morrow Use-case"
        The Morrow criterion accounts for mean stress effects in high-cycle fatigue
        by modifying the stress amplitude based on the true fracture strength.

    ??? abstract "Math Equations"
        The Morrow equivalent stress amplitude is calculated as:

        $$
        \displaystyle\sigma_{aeq}=\frac{\sigma_a}{1-\frac{\sigma_m}{\sigma_{true}} }
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        true_frac_stress: Array-like of true tensile fracture stress. Must be
            broadcastable with stress_amp and mean_stress. Leading dimensions
            are preserved.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress exceeds true fracture stress.
        ValueError: If true fracture stress is not positive.
        ValueError: If mean stress is equal to true fracture stress, resulting in
            infinite equivalent stress amplitude.
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    true_frac_stress_arr = np.asarray(true_frac_stress, dtype=np.float64)

    if np.any(true_frac_stress_arr <= 0):
        raise ValueError("True fracture stress must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = mean_stress_arr / true_frac_stress_arr

    if np.any(ratio == 1.0):
        raise ValueError(
            "Mean stress equals true fracture stress this would result in "
            "infinite equivalent stress amplitude."
        )
    elif np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress exceeds true fracture stress. ",
            UserWarning,
            stacklevel=2,
        )

    return stress_amp_arr / (1 - mean_stress_arr / true_frac_stress_arr)


def calc_stress_eq_amp_soderberg(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    yield_strength: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Soderberg criterion.

    ??? info "Soderberg Use-case"
        The Soderberg criterion accounts for mean stress effects in high-cycle fatigue
        by modifying the stress amplitude based on the yield strength using a linear
        relationship.

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

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress exceeds yield strength.
        ValueError: If yield strength is not positive.
        ValueError: If mean stress is equal to yield strength, resulting in
            infinite equivalent stress amplitude.
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    yield_strength_arr = np.asarray(yield_strength, dtype=np.float64)

    if np.any(yield_strength_arr <= 0):
        raise ValueError("Yield strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = mean_stress_arr / yield_strength_arr
    if np.any(ratio == 1.0):
        raise ValueError(
            "Mean stress equals yield strength this would result in "
            "infinite equivalent stress amplitude."
        )
    elif np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress exceeds yield strength. ",
            UserWarning,
            stacklevel=2,
        )

    return stress_amp_arr / (1 - mean_stress_arr / yield_strength_arr)


def calc_stress_eq_amp_smith(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    ult_tensile_strength: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Smith criterion.

    ??? info "Smith Use-case"
        The Smith criterion accounts for mean stress effects in high-cycle fatigue
        by modifying the stress amplitude based on the ultimate tensile strength.

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
        ult_tensile_strength: Array-like of ultimate tensile strengths. Must be broadcastable
            with stress_amp and mean_stress. Leading dimensions are preserved.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        Warning: If mean stress exceeds ultimate tensile strength.
        ValueError: If ultimate tensile strength is not positive.
        ValueError: If mean stress is equal to ultimate tensile strength, resulting in
            infinite equivalent stress amplitude.

    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    ult_tensile_strength_arr = np.asarray(ult_tensile_strength, dtype=np.float64)

    if np.any(ult_tensile_strength_arr <= 0):
        raise ValueError("Ultimate tensile strength must be positive")

    # Check if mean stress approaches or exceeds material parameter
    ratio = mean_stress_arr / ult_tensile_strength_arr
    if np.any(ratio == 1.0):
        raise ValueError(
            "Mean stress equals ultimate tensile strength this would result in "
            "infinite equivalent stress amplitude."
        )
    elif np.any(ratio > 1.0):
        warnings.warn(
            "Mean stress exceeds ultimate tensile strength. ",
            UserWarning,
            stacklevel=2,
        )
    return (stress_amp_arr * (1 + mean_stress_arr / ult_tensile_strength_arr)) / (
        1 - mean_stress_arr / ult_tensile_strength_arr
    )


def calc_stress_eq_amp_swt(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Smith-Watson-Topper parameter.

    ??? info "SWT Use-case"
        The Smith-Watson-Topper (SWT) parameter accounts for mean stress effects in
        high-cycle fatigue by combining stress amplitude and maximum stress in the cycle

    ??? abstract "Math Equations"
        The SWT equivalent stress amplitude is calculated as:

        $$
        \sigma_{aeq} = \sqrt{\sigma_{a} \cdot (\sigma_{m} + \sigma_{a})} \\
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        ValueError: If stress amplitude is negative.
        ValueError: If the validity condition σₐ > |σₘ| is not satisfied.

    ??? note "Validity Condition"
        The SWT parameter is valid when $\sigma_a > |\sigma_m|$, ensuring that the
        maximum stress in the cycle is positive (tensile). When this condition is
        not met, a ValueError is raised.

    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)

    # Check for negative stress amplitudes
    if np.any(stress_amp_arr < 0):
        raise ValueError("Stress amplitude must be non-negative")

    # Check validity condition: σₐ > |σₘ|
    abs_mean_stress = np.abs(mean_stress_arr)
    invalid_condition = stress_amp_arr <= abs_mean_stress

    if np.any(invalid_condition):
        raise ValueError(
            "Smith-Watson-Topper parameter validity condition (σₐ > |σₘ|) not "
            "satisfied for some data points. The SWT approach may not be "
            "appropriate for compressive-dominated loading conditions."
        )

    return np.sqrt(stress_amp_arr * (mean_stress_arr + stress_amp_arr))


def calc_stress_eq_amp_walker(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    walker_param: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Walker criterion.

    ??? info "Walker Use-case"
        The Walker criterion accounts for mean stress effects in high-cycle fatigue
        by modifying by combining stress amplitude and maximum stress in the cycle and
        utilizing a material specific exponent - the Walker parameter (γ').

    ??? abstract "Math Equations"
        The Walker equivalent stress amplitude is calculated as:

        $$
        \displaystyle\sigma_{aeq}=\left(\sigma_a+\sigma_m\right)^{1-\gamma'} \cdot
            \sigma_a^{\gamma'}
        $$
    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.
        walker_param: Array-like of Walker exponents (γ'). Must be broadcastable
            with stress_amp and mean_stress. Leading dimensions are preserved.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        ValueError: If input arrays cannot be broadcast together or when the
            condition γ' in (0, 1) is not satisfied.
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    walker_param_arr = np.asarray(walker_param, dtype=np.float64)

    # Check validity of Walker parameter: γ' in range (0, 1)
    invalid_condition = (walker_param_arr < 0) | (walker_param_arr > 1)
    if np.any(invalid_condition):
        raise ValueError("Walker parameter (γ') must be in the range (0, 1). ")

    return (stress_amp_arr + mean_stress_arr) ** (
        1 - walker_param_arr
    ) * stress_amp_arr**walker_param_arr
