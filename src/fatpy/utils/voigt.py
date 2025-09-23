"""Helper functions for handling vectors using Voigt notation.

Overview:
    This module provides utilities for converting between Voigt notation vectors
    and their corresponding symmetric 3x3 tensor representations. Voigt notation
    is commonly used in continuum mechanics to represent stress and strain tensors
    in a compact vector form.

Supported Shapes:
    - Input arrays: (..., 6)
        The last axis contains the 6 Voigt components:
            (σ_11, σ_22, σ_33, σ_23, σ_13, σ_12)
            or equivalently (σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy)
    - Output arrays: (..., 3, 3)
        The last two axes represent the symmetric 3x3 tensor for each input index.
        Any number of leading dimensions is supported and preserved.

Usage:
    Use these functions to convert between Voigt vectors and tensor representations
    for stress or strain, ensuring compatibility with numerical routines that expect
    either format.

"""

import numpy as np
from numpy.typing import NDArray

VOIGT_COMPONENTS_COUNT = 6


def check_shape(vector: NDArray[np.float64]) -> None:
    """Validate the Voigt vector shape.

    Args:
        vector: Array with shape (..., 6) where the last dimension has length 6.

    Raises:
        ValueError: If input does not have at least two dimensions or the
            second dimension is not 6.
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
        3x3 tensor for each input index. Trailing dimensions are preserved.
    """
    check_shape(vector)

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
