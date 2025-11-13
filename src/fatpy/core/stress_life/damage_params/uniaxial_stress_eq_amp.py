"""Uniaxial fatigue criteria methods for the stress-life approach.

Contains criteria that address uniaxial high-cycle fatigue by incorporating the mean
stress effect through an equivalent stress amplitude approach. By adjusting the stress
amplitude to account for mean stress influences—using models such as Goodman, Gerber,
or Soderberg—they enable more accurate fatigue life predictions where mean stresses
significantly affect material endurance.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_stress_inputs(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    material_param: ArrayLike | np.float64 | None = None,
    param_name: str = "material parameter",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate stress inputs and material parameters for fatigue calculations.

    Args:
        stress_amp: Stress amplitudes (must be non-negative)
        mean_stress: Mean stresses (can be positive or negative)
        material_param: Material strength parameter (must be positive)
        param_name: Name of material parameter for error messages

    Returns:
        Tuple of validated arrays (stress_amp, mean_stress)

    Raises:
        ValueError: If validation fails
        UserWarning: For questionable but not invalid conditions
    """
    stress_amp_arr = np.asarray(stress_amp, dtype=np.float64)
    mean_stress_arr = np.asarray(mean_stress, dtype=np.float64)
    material_param_arr = (
        None if material_param is None else np.asarray(material_param, dtype=np.float64)
    )

    # Check for negative stress amplitudes
    if np.any(stress_amp_arr < 0):
        raise ValueError("Stress amplitude must be non-negative")

    # Validate material parameter if provided
    if material_param_arr is not None:
        if np.any(material_param_arr <= 0):
            raise ValueError(f"{param_name} must be positive")

        # Check if mean stress approaches or exceeds material parameter
        abs_mean = np.abs(mean_stress_arr)
        ratio = abs_mean / material_param_arr

        if np.any(ratio >= 1.0):
            raise ValueError(
                f"Mean stress magnitude ({np.max(abs_mean):.1f}) exceeds or equals "
                f"{param_name} ({np.min(material_param_arr):.1f}). This would result in"
                " infinite or negative equivalent stress amplitude."
            )

    return stress_amp_arr, mean_stress_arr


def calc_stress_eq_amp_swt(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Smith-Watson-Topper parameter.

    The Smith-Watson-Topper (SWT) parameter accounts for mean stress effects in
    high-cycle fatigue by combining stress amplitude and maximum stress in the cycle.


    ??? abstract "Math Equations"
        The SWT equivalent stress amplitude is calculated as:

        $$
        \sigma_{aeq} = \sqrt{\sigma_{a} \cdot (\sigma_{m} + \sigma_{a})}
        $$

    Args:
        stress_amp: Array-like of stress amplitudes. Leading dimensions are preserved.
        mean_stress: Array-like of mean stresses. Must be broadcastable with
            stress_amp. Leading dimensions are preserved.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        ValueError: If input arrays cannot be broadcast together or when the
            condition σₐ > |σₘ| is not satisfied.

    ??? note "Validity Condition"
        The SWT parameter is valid when $\sigma_a > |\sigma_m|$, ensuring that the
        maximum stress in the cycle is positive (tensile). When this condition is
        not met, a warning is issued as the SWT approach may not be appropriate
        for compressive-dominated loading conditions.

    """
    stress_amp_arr, mean_stress_arr = _validate_stress_inputs(stress_amp, mean_stress)

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


def calc_stress_eq_amp_goodman(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    ult_stress: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Goodman criterion.

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
        ult_stress: Array-like of ultimate tensile strengths. Must be broadcastable
            with stress_amp and mean_stress. Leading dimensions are preserved.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        ValueError: If input arrays cannot be broadcast together.
    """
    stress_amp_arr, mean_stress_arr = _validate_stress_inputs(
        stress_amp, mean_stress, ult_stress, "Ultimate tensile strength"
    )

    ult_stress_arr = np.asarray(ult_stress, dtype=np.float64)

    return stress_amp_arr / (1 - mean_stress_arr / ult_stress_arr)


def calc_stress_eq_amp_gerber(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    ult_stress: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Gerber criterion.

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
        ult_stress: Array-like of ultimate tensile strengths. Must be broadcastable
            with stress_amp and mean_stress. Leading dimensions are preserved.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        ValueError: If input arrays cannot be broadcast together.

    """
    stress_amp_arr, mean_stress_arr = _validate_stress_inputs(
        stress_amp, mean_stress, ult_stress, "Ultimate tensile strength"
    )
    ult_stress_arr = np.asarray(ult_stress, dtype=np.float64)

    return stress_amp_arr / (1 - (mean_stress_arr / ult_stress_arr) ** 2)


def calc_stress_eq_amp_morrow(
    stress_amp: ArrayLike | np.float64,
    mean_stress: ArrayLike | np.float64,
    true_fract_stress: ArrayLike | np.float64,
) -> NDArray[np.float64]:
    r"""Calculate equivalent stress amplitude using Morrow criterion.

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
        true_fract_stress: Array-like of true tensile fracture stress. Must be
            broadcastable with stress_amp and mean_stress. Leading dimensions
            are preserved.

    Returns:
        Array of equivalent stress amplitudes. Shape follows NumPy broadcasting
            rules for the input arrays.

    Raises:
        ValueError: If input arrays cannot be broadcast together
    """
    stress_amp_arr, mean_stress_arr = _validate_stress_inputs(
        stress_amp, mean_stress, true_fract_stress, "True tensile fracture stress"
    )
    true_fract_stress_arr = np.asarray(true_fract_stress, dtype=np.float64)

    return stress_amp_arr / (1 - mean_stress_arr / true_fract_stress_arr)
