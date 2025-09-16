"""Calculate fundamental stress metrics and invariants.

These functions provide principal stresses, maximum shear (Tresca) stress,
hydrostatic stress, von Mises equivalent stress, and invariants of the stress
tensor. They are essential for strength, fatigue, and fracture analyses under
both uniaxial and multiaxial loading conditions.

Conventions:
- Principal stresses are ordered in descending order throughout the module:
    σ1 ≥ σ2 ≥ σ3.
- Principal directions (eigenvectors) are aligned to this ordering
    (columns correspond to σ1, σ2, σ3).

"""

import numpy as np
from numpy.typing import NDArray

from fatpy.utils import voigt


def calc_hydrostatic_stress(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Calculate the hydrostatic (mean normal) stress for each stress state.

    ??? abstract "Math Equations"
        $$
        \sigma_H = \frac{1}{3} tr(\sigma) =
        \frac{1}{3} (\sigma_{11} + \sigma_{22} + \sigma_{33})
        $$

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in
            Voigt notation.

    Returns:
        Array of shape (n,). Hydrostatic stress for each input state.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    voigt.check_shape(stress_voigt)

    return (stress_voigt[:, 0] + stress_voigt[:, 1] + stress_voigt[:, 2]) / 3.0


def calc_principal_stresses_and_directions(
    stress_voigt: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Calculate principal stresses and principal directions for each state.

    ??? abstract "Math Equations"
        Principal stresses and directions are found by solving the eigenvalue problem
        for the stress tensor:

        $$ \sigma \mathbf{v} = \lambda \mathbf{v} $$

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in
            Voigt notation.

    Returns:
        Tuple (eigvals, eigvecs):
        - eigvals: Array of shape (n, 3). Principal stresses (descending: σ1 ≥ σ2 ≥ σ3)
        - eigvecs: Array of shape (n, 3, 3). Principal directions (columns are
            eigenvectors) aligned with eigvals in the same order.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    voigt.check_shape(stress_voigt)

    tensor = voigt.voigt_to_tensor(stress_voigt)
    eigvals, eigvecs = np.linalg.eigh(tensor)  # ascending by default

    # Reorder to descending and keep eigenvectors aligned with eigenvalues
    idx = np.argsort(eigvals, axis=1)[:, ::-1]  # (n,3)
    eigvals_sorted = np.take_along_axis(eigvals, idx, axis=1)
    eigvecs_sorted = np.take_along_axis(eigvecs, idx[:, None, :], axis=2)

    return eigvals_sorted, eigvecs_sorted


def calc_principal_stresses(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Calculate principal stresses for each stress state.

    ??? abstract "Math Equations"
        Principal stresses are found by solving the eigenvalue problem
        for the stress tensor:

        $$ \sigma \mathbf{v} = \lambda \mathbf{v} $$

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in
            Voigt notation.

    Returns:
        Array of shape (n, 3). Principal stresses (descending: σ1 ≥ σ2 ≥ σ3).

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    voigt.check_shape(stress_voigt)

    tensor = voigt.voigt_to_tensor(stress_voigt)
    eigvals = np.linalg.eigvalsh(tensor)

    return np.sort(eigvals, axis=1)[:, ::-1]


def calc_principal_directions(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Calculate principal directions (eigenvectors) for each stress state.

    ??? abstract "Math Equations"
        Principal directions are found by solving the eigenvalue problem
        for the stress tensor:

        $$ \sigma \mathbf{v} = \lambda \mathbf{v} $$

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in
            Voigt notation.

    Returns:
        Array of shape (n, 3, 3). Principal directions (columns are eigenvectors)
            aligned with descending principal stresses: σ1, σ2, σ3.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    voigt.check_shape(stress_voigt)

    # Reuse the ordering logic from the combined function to ensure consistency
    _, eigvecs = calc_principal_stresses_and_directions(stress_voigt)
    return eigvecs


def calc_stress_invariants(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Calculate the first, second, and third invariants for each stress state.

    ??? abstract "Math Equations"
        $$
        \begin{align*}
        I_1 &=  tr(\sigma), \\
        I_2 &= \frac{1}{2}(I_1^{2} - tr(\sigma^{2})), \\
        I_3 &= \det(\sigma)
        \end{align*}
        $$

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in
            Voigt notation.

    Returns:
        Array of shape (n, 3). Columns are (I1, I2, I3) for each row.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    voigt.check_shape(stress_voigt)

    tensor = voigt.voigt_to_tensor(stress_voigt)
    invariant_1 = np.trace(tensor, axis1=1, axis2=2)
    invariant_2 = 0.5 * (
        invariant_1**2 - np.trace(np.matmul(tensor, tensor), axis1=1, axis2=2)
    )
    invariant_3 = np.linalg.det(tensor)

    return np.stack((invariant_1, invariant_2, invariant_3), axis=1)


def calc_von_mises_stress(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Calculate von Mises equivalent stress for each stress state.

    ??? abstract "Math Equations"
        $$
        \sigma_{vM} = \tfrac{\sqrt{2}}{2}\sqrt{
        (\sigma_{11}-\sigma_{22})^2
        +(\sigma_{22}-\sigma_{33})^2
        +(\sigma_{33}-\sigma_{11})^2
        + 3(\sigma_{12}^2+\sigma_{23}^2+\sigma_{13}^2)}
        $$

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in
            Voigt notation.

    Returns:
        Array of shape (n,). Von Mises equivalent stress for each row.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    voigt.check_shape(stress_voigt)

    sx = stress_voigt[:, 0]
    sy = stress_voigt[:, 1]
    sz = stress_voigt[:, 2]
    syz = stress_voigt[:, 3]
    sxz = stress_voigt[:, 4]
    sxy = stress_voigt[:, 5]

    return np.sqrt(
        0.5
        * (
            (sx - sy) ** 2
            + (sy - sz) ** 2
            + (sz - sx) ** 2
            + 6 * (sxy**2 + syz**2 + sxz**2)
        )
    )


def calc_signed_von_mises_by_hydrostatic(
    stress_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate signed von Mises stress for each stress state.

    Sign is determined by the hydrostatic stress.

    ??? abstract "Math Equations"
        $$\sigma_{SvM} = sgn(\sigma_H) \cdot \sigma_{vM}$$

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in
            Voigt notation.

    Returns:
        Array of shape (n,). Signed von Mises stress for each row.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    von_mises = calc_von_mises_stress(stress_voigt)
    sign = np.sign(calc_hydrostatic_stress(stress_voigt))
    return sign * von_mises


def calc_signed_von_mises_by_max_abs_principal(
    stress_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate signed von Mises stress for each stress state.

    Sign is determined by the maximum absolute principal stress value.

    ??? abstract "Math Equations"
        $$\sigma_{SvM} = sgn(\frac{\sigma_{1}+\sigma_{3}}{2}) \cdot \sigma_{vM}$$

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in
            Voigt notation.

    Returns:
        Array of shape (n,). Signed von Mises stress for each row.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    von_mises = calc_von_mises_stress(stress_voigt)
    principals = calc_principal_stresses(stress_voigt)

    avg_13 = 0.5 * (principals[:, 0] + principals[:, 2])
    sign = np.sign(avg_13).astype(np.float64, copy=False)

    return sign * von_mises


def calc_signed_von_mises_by_first_invariant(
    stress_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate signed von Mises stress for each stress state.

    Sign is determined by the first invariant of the stress tensor.

    ??? abstract "Math Equations"
        $$\sigma_{SvM} = sgn(tr(\sigma)) \cdot \sigma_{vM}$$

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in
            Voigt notation.

    Returns:
        Array of shape (n,). Signed von Mises stress for each row.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    von_mises = calc_von_mises_stress(stress_voigt)
    invariant_1 = stress_voigt[:, 0] + stress_voigt[:, 1] + stress_voigt[:, 2]

    sign = np.sign(invariant_1).astype(np.float64, copy=False)

    return sign * von_mises


def calc_tresca_stress(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Calculate Tresca (maximum shear) stress for each stress state.

    ??? abstract "Math Equations"
        $$\sigma_{\tau_{max}} = \frac{\sigma_{1} - \sigma_{3}}{2}$$

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in
            Voigt notation.

    Returns:
        Array of shape (n,). Tresca stress for each row. For principal stresses
        σ1 ≥ σ2 ≥ σ3, Tresca = (σ1 − σ3)/2.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    principals = calc_principal_stresses(stress_voigt)
    return 0.5 * (principals[:, 0] - principals[:, 2])


def calc_signed_tresca_by_hydrostatic(
    stress_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate signed Tresca stress for each stress state.

    Sign is determined by the hydrostatic stress.

    ??? abstract "Math Equations"
        $$\sigma_{S\tau_{max}} = sgn(\sigma_H) \cdot \sigma_{\tau_{max}}$$

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in
            Voigt notation.

    Returns:
        Array of shape (n,). Signed Tresca stress for each row.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    tresca = calc_tresca_stress(stress_voigt)
    sign = np.sign(calc_hydrostatic_stress(stress_voigt))
    return sign * tresca


def calc_signed_tresca_by_max_abs_principal(
    stress_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate signed Tresca stress for each stress state.

    Sign is determined by the maximum absolute principal stress value.

    ??? abstract "Math Equations"
        $$
        \sigma_{S\tau_{max}} = sgn(\frac{\sigma_{1}+\sigma_{3}}{2})
        \cdot \sigma_{\tau_{max}}
        $$

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in
            Voigt notation.

    Returns:
        Array of shape (n,). Signed Tresca stress for each row.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    tresca = calc_tresca_stress(stress_voigt)
    principals = calc_principal_stresses(stress_voigt)

    avg_13 = 0.5 * (principals[:, 0] + principals[:, 2])
    sign = np.sign(avg_13).astype(np.float64, copy=False)

    return sign * tresca


if __name__ == "__main__":
    # Example usage
    stress_example = np.array(
        [
            [10, 20, 30, 0, 0, 0],
            [50, -50, 0, 0, 0, 10],
            [0, 0, 1000, 0, 0, -50],
        ]
    )
    # print("Principal Stresses:", calc_principal_stresses(stress_example))
    # print("Hydrostatic Stress:", calc_hydrostatic_stress(stress_example))
    # print("Von Mises Stress:", calc_von_mises_stress(stress_example))
    # print("signed Von Mises Stress:", calc_signed_von_mises(stress_example))
    # print("Tresca Stress:", calc_tresca_stress(stress_example))
    # print("signed Tresca Stress:", calc_signed_tresca(stress_example))
    # eigvals, eigvecs = calc_principal_stresses_and_directions(stress_example)
    # print("Principal Stresses:", eigvals)
    # print("Principal Directions:", eigvecs)
    # print("Principal Stresses:", eigvals)
    # print("Principal Directions:", eigvecs)
    # print("Principal Directions:", eigvecs)
