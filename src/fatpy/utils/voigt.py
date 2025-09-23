"""Helper functions for handling vectors using Voigt notation.

All functions accept arrays with shape (n, 6, ...) where any trailing
dimensions after the 6-component Voigt axis are allowed. The 3x3 tensor
representation uses the last two axes for matrix indices, i.e.
returned tensors have shape (n, ..., 3, 3).
"""

# TODO: N-Dimensional support for trailing dims
#  - voigt, tensor dimensions have to be the last axes
#  - (..., 6), (..., 3, 3)

import numpy as np
from numpy.typing import NDArray

VOIGT_COMPONENTS_COUNT = 6
MIN_VECTOR_DIMS = 2


def check_shape(vector: NDArray[np.float64]) -> None:
    """Validate the Voigt vector shape.

    Args:
        vector: Array with shape (n, 6, ...) where axis=1 has length 6.

    Raises:
        ValueError: If input does not have at least two dimensions or the
            second dimension is not 6.
    """
    if vector.ndim < MIN_VECTOR_DIMS:
        raise ValueError("Input must be at least a 2D array (n, 6, ...).")

    if vector.shape[1] != VOIGT_COMPONENTS_COUNT:
        raise ValueError(
            "Second dimension must correspond to 6 Voigt "
            "stress/strain components (n, 6, ...)."
        )


def voigt_to_tensor(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert Voigt vectors to symmetric 3x3 tensors.

    Args:
        vector: Array of shape (n, 6, ...) where axis=1 contains the Voigt
            components (σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy).

    Returns:
        Array with shape (n, ..., 3, 3). The last two axes are the symmetric
        3x3 tensor for each input index. Trailing dimensions from the input
        are preserved before the final two matrix axes.
    """
    check_shape(vector)

    # Move the Voigt component axis to the end for easy broadcasting, then
    # construct tensors with matrix axes as the last two dims.
    # Input shape: (n, 6, d1, d2, ...)
    # We'll output shape: (n, d1, d2, ..., 3, 3)
    leading = vector.shape[0]
    trailing = vector.shape[2:]

    out_shape = (leading,) + trailing + (3, 3)
    tensor = np.zeros(out_shape, dtype=vector.dtype)

    # Helper to index into input with preserved trailing dims
    # Normal components
    tensor[(..., 0, 0)] = vector[:, 0].reshape((leading,) + trailing)
    tensor[(..., 1, 1)] = vector[:, 1].reshape((leading,) + trailing)
    tensor[(..., 2, 2)] = vector[:, 2].reshape((leading,) + trailing)

    # Off-diagonal: Voigt indices 3 -> σ_yz, 4 -> σ_xz, 5 -> σ_xy
    tensor[(..., 1, 2)] = vector[:, 3].reshape((leading,) + trailing)
    tensor[(..., 2, 1)] = tensor[(..., 1, 2)]

    tensor[(..., 0, 2)] = vector[:, 4].reshape((leading,) + trailing)
    tensor[(..., 2, 0)] = tensor[(..., 0, 2)]

    tensor[(..., 0, 1)] = vector[:, 5].reshape((leading,) + trailing)
    tensor[(..., 1, 0)] = tensor[(..., 0, 1)]

    return tensor
