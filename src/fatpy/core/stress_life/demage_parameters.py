"""Damage parameters calculation methods for the stress-life."""
import inspect
import warnings
import numpy as np
from numpy.typing import NDArray

def s_eqa_asme(fat_strength_coef: NDArray[np.float64],
                 fat_strength_exp: NDArray[np.float64],
                 yilel_stress: NDArray[np.float64],
                 stress_amp: NDArray[np.float64],
                 mean_stress: NDArray[np.float64]
                 ) -> NDArray[np.uint64]:
    """
    Uniaxial high-cycle fatigue criterion using the equivalent stress amplitude
    based on ASME

    References
    ----------
    [1] J. Papuga, I. Vízková, M. Lutovinov, M. Nesládek: Mean stress effect in
        stress-life fatigue prediction re-evaluated, MATEC Web of Conferences 165,
        10018, 2018.
    
    Parameters
    ----------
    :param fat_strength_coef : NDArray[np.float64] Manson-Coffin and Basquin
        equation fatigue strength coefficient in [MPa]
    :param fat_strength_exp : NDArray[np.float64] Manson-Coffin and Basquin
        equation fatigue strength exponent
    :param yilel_stress : NDArray[np.float64] Tensile yield strength in [MPa]
    :param stress_amp : NDArray[np.float64] Stress amplitude in [MPa]
    :param mean_stress : NDArray[np.float64] Mean stress in [MPa]
    
    :return: NDArray[np.uint64] Estimated repetitions N of a given load cycle
    to failure
    """
    signature = inspect.signature(s_eqa_asme)
    for param in signature.parameters.values():
        if param.name not in locals():
            raise ValueError(f"Missing argument: {param.name}")
        if not isinstance(locals()[param.name], np.ndarray):
            raise TypeError(f"{param.name} must be a numpy ndarray.")

    if np.any(fat_strength_coef <= 0):
        raise ValueError("fat_strength_coef must be positive values.")
    if np.any(fat_strength_exp >= 0):
        raise ValueError("fat_strength_exp must be negative values.")
    if np.any(yilel_stress <= 0):
        raise ValueError("yilel_stress must be positive values.")
    if np.any(stress_amp < 0):
        raise ValueError("stress_amp must be non-negative values.")
    if np.any(mean_stress/yilel_stress >= 1):
        raise ValueError("mean_stress/yilel_stress must be less than 1.")

    stress_aeq = stress_amp / np.sqrt(1 - (mean_stress/yilel_stress)**2)

    if np.any(stress_aeq >= fat_strength_coef):
        warnings.warn("excessive loading detected resulting in N < 1.", UserWarning)

    N = 0.5 * (stress_aeq / fat_strength_coef)**(1/fat_strength_exp)

    return N.astype(np.uint64)
    

