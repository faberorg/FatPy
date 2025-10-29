"""Calculate fundamental strain metrics and invariants.

These functions provide principal strains, hydrostatic strain, von Mises equivalent
strain, and invariants of the strain tensor. They are essential for strength, fatigue,
and fracture analyses under both uniaxial and multiaxial loading conditions.

Conventions:
- Vectors use Voigt notation with shape (..., 6), where the last dimension
  contains the six Voigt components and leading dimensions are preserved:

        (ε_11, ε_22, ε_33, ε_23, ε_13, ε_12)
        (ε_xx, ε_yy, ε_zz, ε_yz, ε_xz, ε_xy)

- Principal strains are ordered in descending order throughout the module:
    ε1 ≥ ε2 ≥ ε3.
- Principal directions (eigenvectors) are aligned to this ordering
    (columns correspond to ε1, ε2, ε3).
"""

import numpy as np
from numpy.typing import NDArray

from fatpy.utils import voigt


def calc_principal_strains_and_directions(
    strain_vector_voigt: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Calculate principal strains and principal directions for each state.

    ??? abstract "Math Equations"
        Principal strains and directions are found by solving the eigenvalue problem
        for the strain tensor:

        $$ \varepsilon \mathbf{v} = \lambda \mathbf{v} $$

    Args:
        strain_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt strain components. Leading dimensions are preserved.

    Returns:
        Tuple (eigvals, eigvecs):
            - eigvals: Array of shape (..., 3). Principal strains
            (descending: ε_1 ≥ ε_2 ≥ ε_3) with leading dimensions preserved.
            - eigvecs: Array of shape (..., 3, 3). Principal directions (columns are
            eigenvectors) aligned with eigvals in the same order. The last two
            dimensions are the 3x3 eigenvector matrix for each input.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(strain_vector_voigt)

    tensor = voigt.voigt_to_tensor(strain_vector_voigt)
    eigvals, eigvecs = np.linalg.eigh(tensor)
    sorted_indices = np.argsort(eigvals, axis=-1)[..., ::-1]
    eigvals_sorted = np.take_along_axis(eigvals, sorted_indices, axis=-1)
    eigvecs_sorted = np.take_along_axis(
        eigvecs, np.expand_dims(sorted_indices, axis=-2), axis=-1
    )

    return eigvals_sorted, eigvecs_sorted


def calc_principal_strains(
    strain_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate principal strains for each strain state.

    ??? abstract "Math Equations"
        Principal strains are found by solving the eigenvalue problem
        for the strain tensor:

        $$ \varepsilon \mathbf{v} = \lambda \mathbf{v} $$

    Args:
        strain_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt strain components. Leading dimensions are preserved.

    Returns:
        Array of shape (..., 3). Principal strains (descending: ε1 ≥ ε2 ≥ ε3).

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(strain_vector_voigt)

    tensor = voigt.voigt_to_tensor(strain_vector_voigt)
    eigvals = np.linalg.eigvalsh(tensor)

    return np.sort(eigvals, axis=-1)[..., ::-1]


def calc_strain_invariants(
    strain_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate the first, second, and third invariants for each strain state.

    ??? abstract "Math Equations"
        $$
        \begin{align*}
        I_1 &=  tr(\varepsilon), \\
        I_2 &= \tfrac{1}{2}\big(I_1^{2} - tr(\varepsilon^{2})\big), \\
        I_3 &= \det(\varepsilon)
        \end{align*}
        $$

    Args:
        strain_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt strain components. Leading dimensions are preserved.

    Returns:
        Array of shape (..., 3). The last dimension contains (I1, I2, I3) for
            each entry.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(strain_vector_voigt)

    tensor = voigt.voigt_to_tensor(strain_vector_voigt)
    invariant_1 = np.trace(tensor, axis1=-2, axis2=-1)
    invariant_2 = 0.5 * (
        invariant_1**2 - np.trace(np.matmul(tensor, tensor), axis1=-2, axis2=-1)
    )
    invariant_3 = np.linalg.det(tensor)

    return np.stack((invariant_1, invariant_2, invariant_3), axis=-1)


def calc_volumetric_strain(
    strain_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate the volumetric (mean normal)strain for each strain state.

    ??? abstract "Math Equations"
        $$ \varepsilon_{vol} = \frac{1}{3} \, tr(\varepsilon) =
           \frac{1}{3}(\varepsilon_{11} + \varepsilon_{22} + \varepsilon_{33})
        $$

    Args:
        strain_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt strain components. Leading dimensions are preserved.

    Returns:
        Array of shape (...). Volumetric (mean normal) strain for each input state.
            Tensor rank is reduced by one.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(strain_vector_voigt)

    return (
        strain_vector_voigt[..., 0]
        + strain_vector_voigt[..., 1]
        + strain_vector_voigt[..., 2]
    ) / 3.0


def calc_deviatoric_strain(
    strain_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Calculate the deviatoric strain for each strain state.

    ??? abstract "Math Equations"
        The strain tensor decomposes as:

        $$ \varepsilon = \varepsilon_{dev} + \varepsilon_{vol} \mathbf{I} $$

        where the deviatoric part is traceless and obtained by subtracting the
        volumetric part from the normal components.

        $$ \varepsilon_{dev} = \varepsilon - \frac{1}{3} tr(\varepsilon) $$

    Args:
        strain_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt strain components. Leading dimensions are preserved.

    Returns:
        Array of shape (..., 6). Deviatoric strain for each input state.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(strain_vector_voigt)

    volumetric = calc_volumetric_strain(strain_vector_voigt)
    deviatoric = strain_vector_voigt.copy()
    deviatoric[..., :3] = deviatoric[..., :3] - volumetric[..., None]

    return deviatoric


# Von Mises functions
def calc_von_mises_strain(
    strain_vector_voigt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Von Mises equivalent strain computed directly from Voigt components.

    ??? abstract "Math Equations"
        $$
        \varepsilon_{vM} = \tfrac{\sqrt{2}}{3}\sqrt{
        (\varepsilon_{11}-\varepsilon_{22})^2
        +(\varepsilon_{22}-\varepsilon_{33})^2
        +(\varepsilon_{33}-\varepsilon_{11})^2
        + 6(\varepsilon_{12}^2+\varepsilon_{23}^2+\varepsilon_{13}^2)}
        $$

    Args:
        strain_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt strain components. Leading dimensions are preserved.

    Returns:
        Array of shape (...). Von Mises equivalent strain for each entry.
            Tensor rank is reduced by one.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(strain_vector_voigt)

    e11 = strain_vector_voigt[..., 0]
    e22 = strain_vector_voigt[..., 1]
    e33 = strain_vector_voigt[..., 2]
    e23 = strain_vector_voigt[..., 3]  # epsilon_23
    e13 = strain_vector_voigt[..., 4]  # epsilon_13
    e12 = strain_vector_voigt[..., 5]  # epsilon_12
    return np.sqrt(
        (2.0 / 9.0)
        * (
            (e11 - e22) ** 2
            + (e22 - e33) ** 2
            + (e33 - e11) ** 2
            + 6.0 * (e12**2 + e23**2 + e13**2)
        )
    )


def calc_signed_von_mises_by_max_abs_principal(
    strain_vector_voigt: NDArray[np.float64],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> NDArray[np.float64]:
    r"""Calculate signed von Mises equivalent strain for each strain state.

    Sign is determined by average of the maximum and minimum principal strains.

    ??? note "Sign Convention"
        The sign assignment follows these rules:

        - **Positive (+)**: When (ε₁ + ε₃)/2 > 0 (tension dominant)
        - **Negative (-)**: When (ε₁ + ε₃)/2 < 0 (compression dominant)
        - **Positive (+)**: When (ε₁ + ε₃)/2 ≈ 0 (within tolerance, default fallback)

    Tolerance parameters ensure numerical stability in edge cases where the
    determining value is very close to zero, preventing erratic sign changes
    that could occur due to floating-point precision limitations.

    ??? abstract "Math Equations"
        $$
        \varepsilon_{SvM} = \begin{cases}
        +\varepsilon_{vM} & \text{if } \frac{\varepsilon_1 + \varepsilon_3}{2} \geq 0 \\
        -\varepsilon_{vM} & \text{if } \frac{\varepsilon_1 + \varepsilon_3}{2} < 0
        \end{cases}
        $$

    Args:
        strain_vector_voigt: Array of shape (..., 6). The last dimension contains the
            Voigt strain components. Leading dimensions are preserved.
        rtol: Relative tolerance for comparing the average of maximum and minimum
                principal strain to zero.
            Default is 1e-5.
        atol: Absolute tolerance for comparing the average of maximum and minimum
              principal strain to zero.
            Default is 1e-8.

    Returns:
        Array of shape (...). Signed von Mises equivalent strain for each entry.
            Tensor rank is reduced by one.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    voigt.check_shape(strain_vector_voigt)

    von_mises = calc_von_mises_strain(strain_vector_voigt)
    principals = calc_principal_strains(strain_vector_voigt)

    avg_13 = 0.5 * (principals[..., 0] + principals[..., 2])
    sign = np.sign(avg_13).astype(np.float64, copy=False)
    sign = np.where(np.isclose(avg_13, 0, rtol=rtol, atol=atol), 1.0, sign)

    return sign * von_mises
