"""Calculate fundamental stress metrics and invariants.

These functions provide principal stresses, maximum shear (Tresca) stress,
hydrostatic stress, von Mises equivalent stress, and invariants of the stress
tensor. They are essential for strength, fatigue, and fracture analyses under
both uniaxial and multiaxial loading conditions.

Conventions:
- Vectors use Voigt notation with shape (n, 6, ...).

    For stress tensors, the six Voigt components are:

        (σ_11, σ_22, σ_33, σ_23, σ_13, σ_12)
        (σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy)

- Principal stresses are ordered in descending order throughout the module
(σ_1 ≥ σ_2 ≥ σ_3).

- Principal directions (eigenvectors) are aligned to this ordering
(columns correspond to σ_1, σ_2, σ_3).

"""

# TODO: N-Dimensional support for trailing dims
#  - voigt, tensor dimensions have to be the last axes
#  - (..., 6), (..., 3, 3)

import numpy as np
from numpy.typing import NDArray

from fatpy.utils import voigt


def calc_principal_stresses_and_directions(
    stress_voigt: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Calculate principal stresses and principal directions for each state.

    ??? abstract "Math Equations"
        Principal stresses and directions are found by solving the eigenvalue problem
        for the stress tensor:

        $$ \sigma \mathbf{v} = \lambda \mathbf{v} $$

    Args:
        stress_voigt: Array of shape (n, 6, ...). The second axis of size 6
            contains the Voigt stress components. Any trailing dimensions are
            preserved and the returned arrays will have those trailing dims.

    Returns:
        Tuple (eigvals, eigvecs):
        - eigvals: Array of shape (n, 3, ...). Principal stresses
            (descending: σ_1 ≥ σ_2 ≥ σ_3) with trailing dims preserved.
        - eigvecs: Array of shape (n, 3, 3, ...). Principal directions (columns are
            eigenvectors) aligned with eigvals in the same order. The last two
            axes of this array are the 3x3 eigenvector matrix for each input.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    voigt.check_shape(stress_voigt)

    tensor = voigt.voigt_to_tensor(stress_voigt)

    # tensor has shape (n, ..., 3, 3). Use eigh over the last two axes.
    eigvals, eigvecs = np.linalg.eigh(tensor)  # eigvals shape: (n, ..., 3)

    # Sort eigenvalues descending along the last axis and reorder eigenvectors
    # accordingly. eigvecs has shape (n, ..., 3, 3) where the last axis is
    # the eigenvector corresponding to the eigenvalue at the same position.
    sorted_indices = np.argsort(eigvals, axis=-1)[..., ::-1]
    eigvals_sorted = np.take_along_axis(eigvals, sorted_indices, axis=-1)

    # For eigvecs we need to take along the last axis (eigenvector index)
    # while preserving the matrix axis. Move axes so we can index easily:
    # eigvecs currently: (..., 3, 3) where the last axis indexes components
    # and the second-last axis indexes eigenvector number. We want to reorder
    # the eigenvector-number axis.
    eigvecs_sorted = np.take_along_axis(
        eigvecs, np.expand_dims(sorted_indices, axis=-2), axis=-1
    )

    return eigvals_sorted, eigvecs_sorted


def calc_principal_stresses(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Calculate principal stresses for each stress state.

    ??? abstract "Math Equations"
        Principal stresses are found by solving the eigenvalue problem
        for the stress tensor:

        $$ \sigma \mathbf{v} = \lambda \mathbf{v} $$

    Args:
        stress_voigt: Array of shape (n, 6, ...). The second dimension of size 6
            contains the Voigt stress components. Additional trailing dimensions
            are supported.

    Returns:
        Array of shape (n, 3, ...). Principal stresses (descending: σ1 ≥ σ2 ≥ σ3).

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    voigt.check_shape(stress_voigt)

    tensor = voigt.voigt_to_tensor(stress_voigt)
    eigvals = np.linalg.eigvalsh(tensor)

    # eigvals shape: (n, ..., 3). Return sorted descending along last axis.
    return np.sort(eigvals, axis=-1)[..., ::-1]


def calc_principal_directions(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Calculate principal directions (eigenvectors) for each stress state.

    ??? abstract "Math Equations"
        Principal directions are found by solving the eigenvalue problem
        for the stress tensor:

        $$ \sigma \mathbf{v} = \lambda \mathbf{v} $$

    Args:
        stress_voigt: Array of shape (n, 6, ...). The second dimension of size 6
            contains the Voigt stress components. Additional trailing dimensions
            are supported.

    Returns:
        Array of shape (n, 3, 3, ...). Principal directions (columns are eigenvectors)
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
        stress_voigt: Array of shape (n, 6, ...). The second dimension of size 6
            contains the Voigt stress components. Additional trailing dimensions
            are supported.

    Returns:
        Array of shape (n, 3, ...). Columns are (I1, I2, I3) for each entry.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    voigt.check_shape(stress_voigt)

    tensor = voigt.voigt_to_tensor(stress_voigt)
    invariant_1 = np.trace(tensor, axis1=-2, axis2=-1)
    invariant_2 = 0.5 * (
        invariant_1**2 - np.trace(np.matmul(tensor, tensor), axis1=-2, axis2=-1)
    )
    invariant_3 = np.linalg.det(tensor)

    return np.stack((invariant_1, invariant_2, invariant_3), axis=-1)


def calc_hydrostatic_stress(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Calculate the hydrostatic (mean normal) stress for each stress state.

    ??? abstract "Math Equations"
        $$
        \sigma_H = \frac{1}{3} tr(\sigma) =
        \frac{1}{3} (\sigma_{11} + \sigma_{22} + \sigma_{33})
        $$

    Args:
        stress_voigt: Array of shape (n, 6, ...). The second dimension of size 6
            contains the Voigt stress components. Additional trailing dimensions
            are supported.

    Returns:
        Array of shape (n, ...). Hydrostatic stress for each input state.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    voigt.check_shape(stress_voigt)

    # Voigt normal components are at axis=1 indices 0,1,2. Preserve trailing dims.
    return (
        stress_voigt[:, 0, ...] + stress_voigt[:, 1, ...] + stress_voigt[:, 2, ...]
    ) / 3.0


def calc_stress_deviator(
    stress_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate stress deviator for each stress state.

    ??? abstract "Math Equations"
        $$ \mathbf{s} = \sigma - \frac{1}{3} tr(\sigma) $$

    Args:
        stress_voigt: Array of shape (n, 6, ...). The second dimension of size 6
            contains the Voigt stress components. Additional trailing dimensions
            are supported.

    Returns:
        Array of shape (n, 6, ...). Stress deviator for each entry.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    voigt.check_shape(stress_voigt)
    hydrostatic = calc_hydrostatic_stress(stress_voigt)

    deviator = stress_voigt.copy()
    # Subtract hydrostatic from the first three Voigt components (axis=1)
    deviator[:, :3, ...] = deviator[:, :3, ...] - hydrostatic[..., None]

    return deviator


# Von Mises functions
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
        stress_voigt: Array of shape (n, 6, ...). The second dimension of size 6
            contains the Voigt stress components. Additional trailing dimensions
            are supported.

    Returns:
        Array of shape (n, ...). Von Mises equivalent stress for each entry.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    voigt.check_shape(stress_voigt)

    sx = stress_voigt[:, 0, ...]
    sy = stress_voigt[:, 1, ...]
    sz = stress_voigt[:, 2, ...]
    syz = stress_voigt[:, 3, ...]
    sxz = stress_voigt[:, 4, ...]
    sxy = stress_voigt[:, 5, ...]

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
        stress_voigt: Array of shape (n, 6, ...). The second dimension of size 6
            contains the Voigt stress components. Additional trailing dimensions
            are supported.

    Returns:
        Array of shape (n, ...). Signed von Mises stress for each entry.

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
        stress_voigt: Array of shape (n, 6, ...). The second dimension of size 6
            contains the Voigt stress components. Additional trailing dimensions
            are supported.

    Returns:
        Array of shape (n, ...). Signed von Mises stress for each entry.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    von_mises = calc_von_mises_stress(stress_voigt)
    principals = calc_principal_stresses(stress_voigt)

    avg_13 = 0.5 * (principals[..., 0] + principals[..., 2])
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
        stress_voigt: Array of shape (n, 6, ...). The second dimension of size 6
            contains the Voigt stress components. Additional trailing dimensions
            are supported.

    Returns:
        Array of shape (n, ...). Signed von Mises stress for each entry.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    von_mises = calc_von_mises_stress(stress_voigt)
    invariant_1 = (
        stress_voigt[:, 0, ...] + stress_voigt[:, 1, ...] + stress_voigt[:, 2, ...]
    )

    sign = np.sign(invariant_1).astype(np.float64, copy=False)

    return sign * von_mises


# Tresca functions
def calc_tresca_stress(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Calculate Tresca (maximum shear) stress for each stress state.

    ??? abstract "Math Equations"
        $$\sigma_{\tau_{max}} = \frac{\sigma_{1} - \sigma_{3}}{2}$$

    Args:
        stress_voigt: Array of shape (n, 6, ...). The second dimension of size 6
            contains the Voigt stress components. Additional trailing dimensions
            are supported.

    Returns:
        Array of shape (n, ...). Tresca stress for each entry. For principal
            stresses σ1 ≥ σ2 ≥ σ3, Tresca = (σ1 − σ3)/2.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    principals = calc_principal_stresses(stress_voigt)
    return 0.5 * (principals[..., 0] - principals[..., 2])


def calc_signed_tresca_by_hydrostatic(
    stress_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate signed Tresca stress for each stress state.

    Sign is determined by the hydrostatic stress.

    ??? abstract "Math Equations"
        $$\sigma_{S\tau_{max}} = sgn(\sigma_H) \cdot \sigma_{\tau_{max}}$$

    Args:
        stress_voigt: Array of shape (n, 6, ...). The second dimension of size 6
            contains the Voigt stress components. Additional trailing dimensions
            are supported.

    Returns:
        Array of shape (n, ...). Signed Tresca stress for each entry.

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
        stress_voigt: Array of shape (n, 6, ...). The second dimension of size 6
            contains the Voigt stress components. Additional trailing dimensions
            are supported.

    Returns:
        Array of shape (n, ...). Signed Tresca stress for each entry.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    tresca = calc_tresca_stress(stress_voigt)
    principals = calc_principal_stresses(stress_voigt)

    avg_13 = 0.5 * (principals[..., 0] + principals[..., 2])
    sign = np.sign(avg_13).astype(np.float64, copy=False)

    return sign * tresca


# The function using np.linalg.eigh(tensor) is compatible with tensors constructed as:
# tensor[:, 0, 0, ...] = vector[:, 0, ...]
# tensor[:, 1, 1, ...] = vector[:, 1, ...]
# tensor[:, 2, 2, ...] = vector[:, 2, ...]
# tensor[:, [1, 2], [2, 1], ...] = vector[:, [3], ...]
# tensor[:, [0, 2], [2, 0], ...] = vector[:, [4], ...]
# tensor[:, [0, 1], [1, 0], ...] = vector[:, [5], ...]
