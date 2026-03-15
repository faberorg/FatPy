"""Damage parameters calculation methods for the energy-life."""

import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import RootResults


def _fun_swt(
    n: float,
    sig_f: float,
    b: float,
    eps_f: float,
    c: float,
    young_modulus: float,
    p_swt: float,
) -> float:
    """Function for root finding in SWT calculation."""
    sol: float = (
        p_swt**2
        - sig_f**2 * (2 * n) ** (2 * b)
        - young_modulus * eps_f * sig_f * (2 * n) ** (b + c)
    )
    return sol


def swt(
    sig_f: float,
    b: float,
    eps_f: float,
    c: float,
    young_modulus: float,
    eps_a: float,
    sig_m: float,
    sig_a: float,
    n_0: float = 1.0,
) -> int:
    """Calculate the number of cycles to failure according to SWT criterion."""
    if sig_a <= np.abs(sig_m):
        raise ValueError("SWT is only valid for sig_a > |sig_m|.")

    p_swt: float = np.sqrt(young_modulus * eps_a * (sig_m + sig_a))

    solution: RootResults = root_scalar(
        _fun_swt,
        args=(sig_f, b, eps_f, c, young_modulus, p_swt),
        x0=n_0,
        method="newton",
    )

    if not solution.converged:
        raise ValueError("SWT calculation did not converge.")

    return int(solution.root)
