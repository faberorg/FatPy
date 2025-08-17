"""Helper functions for handling vectors using Voigt notation."""

import numpy as np
from numpy.typing import NDArray


def check_shape(voigt_vec: NDArray[np.float64]) -> None:
    """Check the shape of the input Voigt array.

    Args:
        voigt_vec: Array of shape (n, 6). Each row is a vector in Voigt notation.

    Raises:
        ValueError: If input is not a 2D array with 6 columns.
    """
    if voigt_vec.ndim != 2 or voigt_vec.shape[1] != 6:
        raise ValueError("Input must be a n x 6 matrix in Voigt notation.")


def voigt_to_tensor(voigt_vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert Voigt vectors to symmetric 3x3 tensors.

    Args:
        voigt_vec: Array of shape (n, 6). Each row is a vector in Voigt notation.
            (σ11, σ22, σ33, σ23, σ13, σ12)

    Returns:
        Array of shape (n, 3, 3). Symmetric tensors.
    """
    n = voigt_vec.shape[0]
    tensor = np.zeros((n, 3, 3), dtype=np.float64)
    tensor[:, 0, 0] = voigt_vec[:, 0]
    tensor[:, 1, 1] = voigt_vec[:, 1]
    tensor[:, 2, 2] = voigt_vec[:, 2]
    tensor[:, 1, 2] = tensor[:, 2, 1] = voigt_vec[:, 3]
    tensor[:, 0, 2] = tensor[:, 2, 0] = voigt_vec[:, 4]
    tensor[:, 0, 1] = tensor[:, 1, 0] = voigt_vec[:, 5]
    return tensor
