"""Damage parameters calculation methods for the stress-life."""

import numpy as np
from numpy.typing import NDArray


def calc_stress_eq_amp_asme(
    yield_stress: NDArray[np.float64],
    stress_amp: NDArray[np.float64],
    mean_stress: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Stress-life prediction function based on the ASME mean-stress correction.

    Commputes the value of equivalent stress amplitude, :math:`\sigma_{aeq}` in MPa
    for given stress values :math:`\sigma_a` and :math:`\sigma_m` representing a
    single load cycle, using the ASME mean stress correction.

    ### Mathematical formulation:

    $$  \displaystyle\sigma_{aeq}=\frac{\sigma_a}{\left[1-\left(\frac{\sigma_m}{R_e}
    \right)^2\right]^{1/2} } $$

    Args:
        yield_stress: :math:`R_e` - Tensile yield strength in [MPa].
        stress_amp: :math:`\\sigma_a` - Stress amplitude in [MPa].
        mean_stress: :math:`\\sigma_m` - Mean stress in [MPa].

    Returns:
        Equivalent stress amplitude by ASME

    References:
        [1] J. Papuga, I. Vízková, M. Lutovinov, M. Nesládek: Mean stress effect in
        stress-life fatigue prediction re-evaluated, MATEC Web of Conferences 165,
        10018, 2018.
    """
    stress_aeq = stress_amp / np.sqrt(1 - (mean_stress / yield_stress) ** 2)

    return stress_aeq
