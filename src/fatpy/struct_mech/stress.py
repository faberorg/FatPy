"""Calculate fundamental stress metrics and invariants.

These tools provide principal stresses, maximum shear stress, hydrostatic stress,
von Mises equivalent stress, and invariants of the stress tensor. They are
essential for strength, fatigue, and fracture analyses under both uniaxial and
multiaxial loading conditions.

"""

import numpy as np
from numpy.typing import NDArray


def calc_hydrostatic_stress(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the hydrostatic (mean normal) stress for each stress state.

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in Voigt
         notation.

    Returns:
        Array of shape (n,). Hydrostatic stress for each input state.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    if stress_voigt.ndim != 2 or stress_voigt.shape[1] != 6:
        raise ValueError("Input must be a n x 6 matrix in Voigt notation.")

    return (stress_voigt[:, 0] + stress_voigt[:, 1] + stress_voigt[:, 2]) / 3.0


def _voigt_to_tensor(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert Voigt stress vectors to symmetric 3x3 tensors.

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in Voigt
         notation.

    Returns:
        Array of shape (n, 3, 3). Symmetric stress tensors.
    """
    n = stress_voigt.shape[0]
    tensor = np.zeros((n, 3, 3), dtype=np.float64)
    tensor[:, 0, 0] = stress_voigt[:, 0]
    tensor[:, 1, 1] = stress_voigt[:, 1]
    tensor[:, 2, 2] = stress_voigt[:, 2]
    tensor[:, 1, 2] = tensor[:, 2, 1] = stress_voigt[:, 3]
    tensor[:, 0, 2] = tensor[:, 2, 0] = stress_voigt[:, 4]
    tensor[:, 0, 1] = tensor[:, 1, 0] = stress_voigt[:, 5]
    return tensor


def calc_principal_stresses_and_directions(
    stress_voigt: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute principal stresses and principal directions for each stress state.

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in Voigt
         notation.

    Returns:
        Tuple (eigvals, eigvecs):
        - eigvals: Array of shape (n, 3). Principal stresses (ascending order).
        - eigvecs: Array of shape (n, 3, 3). Principal directions (columns are
          eigenvectors).

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    if stress_voigt.ndim != 2 or stress_voigt.shape[1] != 6:
        raise ValueError("Input must be a n x 6 matrix in Voigt notation.")

    tensor = _voigt_to_tensor(stress_voigt)
    return np.linalg.eigh(tensor)


def calc_principal_stresses(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute principal stresses for each stress state.

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in Voigt
         notation.

    Returns:
        Array of shape (n, 3). Principal stresses (sorted descending for each row).

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    if stress_voigt.ndim != 2 or stress_voigt.shape[1] != 6:
        raise ValueError("Input must be a n x 6 matrix in Voigt notation.")

    tensor = _voigt_to_tensor(stress_voigt)
    eigvals = np.linalg.eigvalsh(tensor)

    return np.sort(eigvals, axis=1)[:, ::-1]


def calc_principal_directions(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute principal directions (eigenvectors) for each stress state.

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in Voigt
         notation.

    Returns:
        Array of shape (n, 3, 3). Principal directions (columns are eigenvectors).

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    if stress_voigt.ndim != 2 or stress_voigt.shape[1] != 6:
        raise ValueError("Input must be a n x 6 matrix in Voigt notation.")

    tensor = _voigt_to_tensor(stress_voigt)
    _, eigvecs = np.linalg.eigh(tensor)

    return eigvecs


def calc_stress_invariants(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the first, second, and third invariants for each stress state.

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in Voigt
         notation.

    Returns:
        Array of shape (n, 3). Columns are (I1, I2, I3) for each row.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    if stress_voigt.ndim != 2 or stress_voigt.shape[1] != 6:
        raise ValueError("Input must be a n x 6 matrix in Voigt notation.")

    tensor = _voigt_to_tensor(stress_voigt)
    i1 = np.trace(tensor, axis1=1, axis2=2)
    i2 = 0.5 * (i1**2 - np.trace(np.matmul(tensor, tensor), axis1=1, axis2=2))
    i3 = np.linalg.det(tensor)

    return np.stack((i1, i2, i3), axis=1)


def calc_von_mises_stress(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute von Mises equivalent stress for each stress state.

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in Voigt
         notation.

    Returns:
        Array of shape (n,). Von Mises equivalent stress for each row.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    if stress_voigt.ndim != 2 or stress_voigt.shape[1] != 6:
        raise ValueError("Input must be a n x 6 matrix in Voigt notation.")

    sx, sy, sz, syz, sxz, sxy = stress_voigt.T

    return np.sqrt(  # type: ignore[no-any-return]
        0.5
        * (
            (sx - sy) ** 2
            + (sy - sz) ** 2
            + (sz - sx) ** 2
            + 6 * (sxy**2 + syz**2 + sxz**2)
        )
    )


def calc_signed_von_mises(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute signed von Mises stress for each stress state.

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in Voigt
         notation.

    Returns:
        Array of shape (n,). Signed von Mises stress for each row (sign from hydrostatic
         part).

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    vm = calc_von_mises_stress(stress_voigt)
    sign = np.sign(calc_hydrostatic_stress(stress_voigt))
    return sign * vm


def calc_tresca_stress(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Tresca (maximum shear) stress for each stress state.

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in Voigt
         notation.

    Returns:
        Array of shape (n,). Tresca stress for each row.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    principal = calc_principal_stresses(stress_voigt)
    return 0.5 * (principal[:, 0] - principal[:, 2])


def calc_signed_tresca(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute signed Tresca stress for each stress state.

    Args:
        stress_voigt: Array of shape (n, 6). Each row is a stress vector in Voigt
         notation.

    Returns:
        Array of shape (n,). Signed Tresca stress for each row (sign from hydrostatic
         part).

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    t = calc_tresca_stress(stress_voigt)
    sign = np.sign(calc_hydrostatic_stress(stress_voigt))
    return sign * t


if __name__ == "__main__":
    # Example usage
    stress_example = np.array(
        [
            [100, 50, 25, 0, 0, 0],
            [-50, 50, 0, 0, 0, 0],
            [0, 0, 0, 50, 0, 0],
            [100, 0, 0, 0, 0, 0],
        ]
    )
    print("Stress:", stress_example)
    # print("Principal Stresses:", calc_principal_stresses(stress_example))
    # print("Hydrostatic Stress:", calc_hydrostatic_stress(stress_example))
    # print("Von Mises Stress:", calc_von_mises_stress(stress_example))
    # print("Tresca Stress:", calc_tresca_stress(stress_example))

    eigvals, eigvecs = calc_principal_stresses_and_directions(stress_example)

    print("Principal Stresses:", eigvals)
    print("Principal Directions:", eigvecs)
