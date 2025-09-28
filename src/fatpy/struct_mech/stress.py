"""Calculate fundamental stress metrics and invariants.

These functions provide principal stresses, maximum shear (Tresca) stress,
hydrostatic stress, von Mises equivalent stress, and invariants of the stress
tensor. They are essential for strength, fatigue, and fracture analyses under
both uniaxial and multiaxial loading conditions.

Conventions:
- Vectors use Voigt notation with shape (..., 6), where the last dimension
  contains the six Voigt components and leading dimensions are preserved:

        (σ_11, σ_22, σ_33, σ_23, σ_13, σ_12)
        (σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy)

- Principal stresses are ordered in descending order throughout the module
  (σ_1 ≥ σ_2 ≥ σ_3).

- Principal directions (eigenvectors) are aligned to this ordering
  (columns correspond to σ_1, σ_2, σ_3).

"""

# TODO: implement atol, rtol to signed stress functions

import numpy as np
from numpy.typing import NDArray

from fatpy.utils import voigt


def calc_principal_stresses_and_directions(
    stress_vector_voigt: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Calculate principal stresses and principal directions for each state.

    ??? abstract "Math Equations"
        Principal stresses and directions are found by solving the eigenvalue problem
        for the stress tensor:

        $$ \sigma \mathbf{v} = \lambda \mathbf{v} $$

    Args:
        stress_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt stress components. Leading dimensions are preserved.

    Returns:
        Tuple (eigvals, eigvecs):
        - eigvals: Array of shape (..., 3). Principal stresses
          (descending: σ_1 ≥ σ_2 ≥ σ_3) with leading dimensions preserved.
        - eigvecs: Array of shape (..., 3, 3). Principal directions (columns are
          eigenvectors) aligned with eigvals in the same order. The last two
          dimensions are the 3x3 eigenvector matrix for each input.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(stress_vector_voigt)

    tensor = voigt.voigt_to_tensor(stress_vector_voigt)
    eigvals, eigvecs = np.linalg.eigh(tensor)
    sorted_indices = np.argsort(eigvals, axis=-1)[..., ::-1]
    eigvals_sorted = np.take_along_axis(eigvals, sorted_indices, axis=-1)
    eigvecs_sorted = np.take_along_axis(
        eigvecs, np.expand_dims(sorted_indices, axis=-2), axis=-1
    )

    return eigvals_sorted, eigvecs_sorted


def calc_principal_stresses(
    stress_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate principal stresses for each stress state.

    ??? abstract "Math Equations"
        Principal stresses are found by solving the eigenvalue problem
        for the stress tensor:

        $$ \sigma \mathbf{v} = \lambda \mathbf{v} $$

    Args:
        stress_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt stress components. Leading dimensions are preserved.

    Returns:
        Array of shape (..., 3). Principal stresses (descending: σ1 ≥ σ2 ≥ σ3).

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(stress_vector_voigt)

    tensor = voigt.voigt_to_tensor(stress_vector_voigt)
    eigvals = np.linalg.eigvalsh(tensor)

    # eigvals shape: (n, ..., 3). Return sorted descending along last axis.
    return np.sort(eigvals, axis=-1)[..., ::-1]


def calc_principal_directions(
    stress_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate principal directions (eigenvectors) for each stress state.

    ??? abstract "Math Equations"
        Principal directions are found by solving the eigenvalue problem
        for the stress tensor:

        $$ \sigma \mathbf{v} = \lambda \mathbf{v} $$

    Args:
        stress_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt stress components. Leading dimensions are preserved.

    Returns:
        Array of shape (..., 3, 3). Principal directions (columns are eigenvectors)
            aligned with descending principal stresses: σ1, σ2, σ3.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(stress_vector_voigt)

    # Reuse the ordering logic from the combined function to ensure consistency
    _, eigvecs = calc_principal_stresses_and_directions(stress_vector_voigt)
    return eigvecs


def calc_stress_invariants(
    stress_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
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
        stress_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt stress components. Leading dimensions are preserved.

    Returns:
        Array of shape (..., 3). The last dimension contains (I1, I2, I3) for
            each entry.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(stress_vector_voigt)

    tensor = voigt.voigt_to_tensor(stress_vector_voigt)
    invariant_1 = np.trace(tensor, axis1=-2, axis2=-1)
    invariant_2 = 0.5 * (
        invariant_1**2 - np.trace(np.matmul(tensor, tensor), axis1=-2, axis2=-1)
    )
    invariant_3 = np.linalg.det(tensor)

    return np.stack((invariant_1, invariant_2, invariant_3), axis=-1)


def calc_hydrostatic_stress(
    stress_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate the hydrostatic (mean normal) stress for each stress state.

    ??? abstract "Math Equations"
        $$
        \sigma_H = \frac{1}{3} tr(\sigma) =
        \frac{1}{3} (\sigma_{11} + \sigma_{22} + \sigma_{33})
        $$

    Args:
        stress_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt stress components. Leading dimensions are preserved.

    Returns:
        Array of shape (...). Hydrostatic stress for each input state.
            Tensor rank is reduced by one.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(stress_vector_voigt)

    # Voigt normal components are at last axis indices 0,1,2
    return (
        stress_vector_voigt[..., 0]
        + stress_vector_voigt[..., 1]
        + stress_vector_voigt[..., 2]
    ) / 3.0


def calc_stress_deviator(
    stress_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate stress deviator for each stress state.

    ??? abstract "Math Equations"
        $$ \mathbf{s} = \sigma - \frac{1}{3} tr(\sigma) $$

    Args:
        stress_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt stress components. Leading dimensions are preserved.

    Returns:
        Array of shape (..., 6). Stress deviator for each entry.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(stress_vector_voigt)
    hydrostatic = calc_hydrostatic_stress(stress_vector_voigt)

    deviator = stress_vector_voigt.copy()
    # Subtract hydrostatic from the first three Voigt components (last axis)
    deviator[..., :3] = deviator[..., :3] - hydrostatic[..., None]

    return deviator


# Von Mises functions
def calc_von_mises_stress(
    stress_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
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
        stress_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt stress components. Leading dimensions are preserved.

    Returns:
        Array of shape (...). Von Mises equivalent stress for each entry.
            Tensor rank is reduced by one.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(stress_vector_voigt)

    sx = stress_vector_voigt[..., 0]
    sy = stress_vector_voigt[..., 1]
    sz = stress_vector_voigt[..., 2]
    syz = stress_vector_voigt[..., 3]
    sxz = stress_vector_voigt[..., 4]
    sxy = stress_vector_voigt[..., 5]

    # Von Mises formula expanded to simplify computation
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
    stress_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate signed von Mises stress for each stress state.

    Sign is determined by the hydrostatic stress.

    ??? abstract "Math Equations"
        $$\sigma_{SvM} = sgn(\sigma_H) \cdot \sigma_{vM}$$

    Args:
        stress_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt stress components. Leading dimensions are preserved.

    Returns:
        Array of shape (...). Signed von Mises stress for each entry.
            Tensor rank is reduced by one.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    von_mises = calc_von_mises_stress(stress_vector_voigt)
    hydrostatic_stress = calc_hydrostatic_stress(stress_vector_voigt)

    sign = np.sign(hydrostatic_stress).astype(np.float64, copy=False)
    sign[np.isclose(hydrostatic_stress, 0)] = 1.0

    return sign * von_mises


def calc_signed_von_mises_by_max_abs_principal(
    stress_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate signed von Mises stress for each stress state.

    Sign is determined by average of the maximum and minimum principal stresses.
    In case the maximum absolute principal stress is zero, the sign is set to +1.
    In case the maximum absolute principal stress is equal to negative value of
    the minimum principal stress, the sign is set to +1 as well.

    ??? abstract "Math Equations"
        $$\sigma_{SvM} = sgn(\sigma_{max,abs}) \cdot \sigma_{vM}$$

    Args:
        stress_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt stress components. Leading dimensions are preserved.

    Returns:
        Array of shape (...). Signed von Mises stress for each entry.
            Tensor rank is reduced by one.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    von_mises = calc_von_mises_stress(stress_voigt)
    principals = calc_principal_stresses(stress_voigt)
    avg_13 = 0.5 * (principals[..., 0] + principals[..., 2])
    sign = np.sign(avg_13).astype(np.float64, copy=False)
    sign[np.isclose(avg_13, 0)] = 1.0
    return sign * von_mises


def calc_signed_von_mises_by_first_invariant(
    stress_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate signed von Mises stress for each stress state.

    Sign is determined by the first invariant of the stress tensor.

    ??? abstract "Math Equations"
        $$\sigma_{SvM} = sgn(tr(\sigma)) \cdot \sigma_{vM}$$

    Args:
        stress_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt stress components. Leading dimensions are preserved.

    Returns:
        Array of shape (...). Signed von Mises stress for each entry.
            Tensor rank is reduced by one.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    von_mises = calc_von_mises_stress(stress_vector_voigt)
    invariant_1 = (
        stress_vector_voigt[..., 0]
        + stress_vector_voigt[..., 1]
        + stress_vector_voigt[..., 2]
    )

    sign = np.sign(invariant_1).astype(np.float64, copy=False)
    sign[np.isclose(invariant_1, 0)] = 1.0

    return sign * von_mises


# Tresca functions
def calc_tresca_stress(stress_vector_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Calculate Tresca (maximum shear) stress for each stress state.

    ??? abstract "Math Equations"
        $$\sigma_{\tau_{max}} = \frac{\sigma_{1} - \sigma_{3}}{2}$$

    Args:
        stress_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt stress components. Leading dimensions are preserved.

    Returns:
        Array of shape (...). Tresca stress for each entry. For principal
            stresses σ1 ≥ σ2 ≥ σ3, Tresca = (σ1 − σ3)/2.
            Tensor rank is reduced by one.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    principals = calc_principal_stresses(stress_vector_voigt)
    return 0.5 * (principals[..., 0] - principals[..., 2])


def calc_signed_tresca_by_hydrostatic(
    stress_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate signed Tresca stress for each stress state.

    Sign is determined by the hydrostatic stress.

    ??? abstract "Math Equations"
        $$\sigma_{S\tau_{max}} = sgn(\sigma_H) \cdot \sigma_{\tau_{max}}$$

    Args:
        stress_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt stress components. Leading dimensions are preserved.

    Returns:
        Array of shape (...). Signed Tresca stress for each entry.
            Tensor rank is reduced by one.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    tresca = calc_tresca_stress(stress_vector_voigt)
    hydrostatic_stress = calc_hydrostatic_stress(stress_vector_voigt)

    sign = np.sign(hydrostatic_stress)
    sign[np.isclose(hydrostatic_stress, 0)] = 1.0

    return sign * tresca


def calc_signed_tresca_by_max_abs_principal(
    stress_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate signed Tresca stress for each stress state.

    Sign is determined by the maximum absolute principal stress value.

    ??? abstract "Math Equations"
        $$
        \sigma_{S\tau_{max}} = sgn(\frac{\sigma_{1}+\sigma_{3}}{2})
        \cdot \sigma_{\tau_{max}}
        $$

    Args:
        stress_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt stress components. Leading dimensions are preserved.

    Returns:
        Array of shape (...). Signed Tresca stress for each entry.
            Tensor rank is reduced by one.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    tresca = calc_tresca_stress(stress_vector_voigt)
    principals = calc_principal_stresses(stress_vector_voigt)

    avg_13 = 0.5 * (principals[..., 0] + principals[..., 2])
    sign = np.sign(avg_13).astype(np.float64, copy=False)
    sign[np.isclose(avg_13, 0)] = 1.0

    return sign * tresca
