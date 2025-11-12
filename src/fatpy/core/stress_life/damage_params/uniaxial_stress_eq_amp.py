"""Uniaxial fatigue criteria methods for the stress-life approach.

Contains criteria that address uniaxial high-cycle fatigue by incorporating the mean
stress effect through an equivalent stress amplitude approach. By adjusting the stress
amplitude to account for mean stress influences—using models such as Goodman, Gerber,
or Soderberg—they enable more accurate fatigue life predictions where mean stresses
significantly affect material endurance.
"""

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


def calc_stress_eq_amp_swt(
    stress_amp: ArrayLike | float,
    mean_stress: ArrayLike | float,
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
            rules for the input arrays. Tensor rank matches the broadcasted result
            of the input arrays.

    Raises:
        ValueError: If input arrays cannot be broadcast together.

        UserWarning: When the condition σₐ > |σₘ| is not satisfied.

    ??? note "Validity Condition"
        The SWT parameter is valid when $\sigma_a > |\sigma_m|$, ensuring that the
        maximum stress in the cycle is positive (tensile). When this condition is
        not met, a warning is issued as the SWT approach may not be appropriate
        for compressive-dominated loading conditions.

    """
    stress_amp = np.asarray(stress_amp)
    mean_stress = np.asarray(mean_stress)

    # Check validity condition: σₐ > |σₘ|
    abs_mean_stress = np.abs(mean_stress)
    invalid_condition = stress_amp <= abs_mean_stress

    if np.any(invalid_condition):
        warnings.warn(
            "Smith-Watson-Topper parameter validity condition (σₐ > |σₘ|) not "
            "satisfied for some data points. The SWT approach may not be "
            "appropriate for compressive-dominated loading conditions.",
            UserWarning,
        )
        return

    stress_eq_amp = np.sqrt(stress_amp * (mean_stress + stress_amp))
    return stress_eq_amp
