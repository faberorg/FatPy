"""Tools for working with Voigt notation in continuum mechanics.

Overview:
    This module provides general utilities for handling vectors and tensors
    represented in Voigt notation. Voigt notation is widely used to express
    symmetric 3x3 tensors, such as stress and strain, in a compact vector form.

Input Shape Convention:
    - Arrays are expected to have shape (..., 6), where the last dimension
      contains the six Voigt components:
          (σ_11, σ_22, σ_33, σ_23, σ_13, σ_12)
          (σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy)
      Leading dimensions (the '...' part) are preserved and can be arbitrary,
      allowing batch processing of multiple vectors or tensors.

"""

import numpy as np
from numpy.typing import NDArray

VOIGT_COMPONENTS_COUNT = 6


def check_shape(vector: NDArray[np.float64]) -> None:
    """Validate the Voigt vector shape.

    Args:
        vector: Array with shape (..., 6) where the last dimension has length 6.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    if vector.shape[-1] != VOIGT_COMPONENTS_COUNT:
        raise ValueError(
            "Last dimension must correspond to 6 Voigt "
            "stress/strain components (..., 6)."
        )


def voigt_to_tensor(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert Voigt vectors to symmetric 3x3 tensors.

    Args:
        vector: Array of shape (..., 6) where the last dimension contains the
            stress/strain components in order according to Voigt notation:
                (σ_11, σ_22, σ_33, σ_23, σ_13, σ_12)
                (σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy)

    Returns:
        Array with shape (..., 3, 3). The last two axes are the symmetric
        3x3 tensor for each input index. Leading dimensions are preserved.

    Raises:
        ValueError: If the last dimension is not of size 6.
    """
    check_shape(vector)

    # (1e6, 50, 8, 6) -> (1e6, 50, 8) + (3, 3) -> (1e6, 50, 8, 3, 3)
    shape = vector.shape[:-1] + (3, 3)
    tensor = np.zeros(shape, dtype=vector.dtype)

    # Normal components
    tensor[(..., 0, 0)] = vector[..., 0]  # xx
    tensor[(..., 1, 1)] = vector[..., 1]  # yy
    tensor[(..., 2, 2)] = vector[..., 2]  # zz

    # Shear components
    tensor[(..., [1, 2], [2, 1])] = vector[..., [3]]  # yz
    tensor[(..., [0, 2], [2, 0])] = vector[..., [4]]  # xz
    tensor[(..., [0, 1], [1, 0])] = vector[..., [5]]  # xy

    return tensor


def tensor_to_voigt(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert symmetric 3x3 tensors to Voigt vectors.

    Args:
        tensor: Array of shape (..., 3, 3) where the last two dimensions
            represent symmetric 3x3 tensors.

    Returns:
        Array with shape (..., 6). The last dimension contains the stress/strain
        components in order according to Voigt notation:
            (σ_11, σ_22, σ_33, σ_23, σ_13, σ_12)
            (σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy)
        Leading dimensions are preserved.

    Raises:
        ValueError: If the last two dimensions are not of size (3, 3).
    """
    if tensor.shape[-2:] != (3, 3):
        raise ValueError("Last two dimensions must correspond to (3, 3) tensors.")

    shape = tensor.shape[:-2] + (VOIGT_COMPONENTS_COUNT,)
    array = np.zeros(shape, dtype=tensor.dtype)

    # Normal components
    array[..., 0] = tensor[..., 0, 0]  # xx
    array[..., 1] = tensor[..., 1, 1]  # yy
    array[..., 2] = tensor[..., 2, 2]  # zz

    # Shear components
    array[..., 3] = tensor[..., 1, 2]  # yz
    array[..., 4] = tensor[..., 0, 2]  # xz
    array[..., 5] = tensor[..., 0, 1]  # xy

    return array
