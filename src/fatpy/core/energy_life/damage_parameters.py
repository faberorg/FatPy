"""Damage parameters calculation methods for the energy-life."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar
from scipy.optimize import RootResults


def calc_life_swt(
    N: NDArray[np.float64],  # number of cycles
    fat_strength_coef: NDArray[np.float64],
    fat_strength_exp: NDArray[np.float64],
    fat_ductility_coef: NDArray[np.float64],
    fat_ductility_exp: NDArray[np.float64],
    young_modulus: NDArray[np.float64],
    p_swt: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Function for root finding in SWT calculation."""
    sol: NDArray[np.float64] = (
        p_swt**2
        - fat_strength_coef**2 * (2 * N) ** (2 * fat_strength_exp)
        - young_modulus
        * fat_ductility_coef
        * fat_strength_coef
        * (2 * N) ** (fat_strength_exp + fat_ductility_exp)
    )
    return sol


def calc_dmg_param_swt(
    elastic_modulus: NDArray[np.float64],
    strain_amp: NDArray[np.float64],
    mean_stress: NDArray[np.float64],
    stress_amp: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the SWT damage parameter."""
    p_swt: NDArray[np.float64] = np.sqrt(
        elastic_modulus * strain_amp * (mean_stress + stress_amp)
    )
    return p_swt


# def swt(
#     en_curve_parameters: dict[str, NDArray[np.float64]],
#     stress_strain_values: dict[str, NDArray[np.float64]],
#     N_0: float = 1.0,
# ) -> int:
#     """Calculate the number of cycles to failure according to SWT criterion."""
#     n_values = swt_array(en_curve_parameters, stress_strain_values, N_0=N_0)
#     return int(n_values.item())


def swt(
    en_curve_parameters: dict[str, NDArray[np.float64]],
    stress_strain_values: dict[str, NDArray[np.float64]],
    N_0: float = 1.0,
) -> NDArray[np.int64]:
    """Calculate SWT cycles to failure for array-like inputs.

    All input values are broadcast to a common shape and solved elementwise.
    """
    fat_strength_coef = np.asarray(
        en_curve_parameters["fat_strength_coef"], dtype=np.float64
    )
    fat_strength_exp = np.asarray(
        en_curve_parameters["fat_strength_exp"], dtype=np.float64
    )
    fat_ductility_coef = np.asarray(
        en_curve_parameters["fat_ductility_coef"], dtype=np.float64
    )
    fat_ductility_exp = np.asarray(
        en_curve_parameters["fat_ductility_exp"], dtype=np.float64
    )
    elastic_modulus = np.asarray(
        en_curve_parameters["elastic_modulus"], dtype=np.float64
    )

    strain_amp = np.asarray(stress_strain_values["strain_amp"], dtype=np.float64)
    mean_stress = np.asarray(stress_strain_values["mean_stress"], dtype=np.float64)
    stress_amp = np.asarray(stress_strain_values["stress_amp"], dtype=np.float64)

    (
        fat_strength_coef,
        fat_strength_exp,
        fat_ductility_coef,
        fat_ductility_exp,
        elastic_modulus,
        strain_amp,
        mean_stress,
        stress_amp,
    ) = np.broadcast_arrays(
        fat_strength_coef,
        fat_strength_exp,
        fat_ductility_coef,
        fat_ductility_exp,
        elastic_modulus,
        strain_amp,
        mean_stress,
        stress_amp,
    )

    if np.any(stress_amp <= np.abs(mean_stress)):
        raise ValueError("SWT is only valid for stress_amp > |mean_stress|.")

    p_swt = calc_dmg_param_swt(elastic_modulus, strain_amp, mean_stress, stress_amp)
    roots = np.empty_like(p_swt, dtype=np.float64)

    for idx in np.ndindex(p_swt.shape):
        solution: RootResults = root_scalar(
            calc_life_swt,
            args=(
                fat_strength_coef[idx],
                fat_strength_exp[idx],
                fat_ductility_coef[idx],
                fat_ductility_exp[idx],
                elastic_modulus[idx],
                p_swt[idx],
            ),
            x0=N_0,
            method="newton",
        )
        if not solution.converged:
            raise ValueError(f"SWT calculation did not converge at index {idx}.")
        roots[idx] = solution.root

    return roots.astype(np.int64)
