import numpy as np
from typing import Union, Sequence
from numpy.typing import NDArray


NumericInput = Union[float, int]
ArrayLike = Union[Sequence[NumericInput], NDArray[np.float64], NumericInput]


def von_mises_stress(
    s11: ArrayLike,
    s22: ArrayLike,
    s33: ArrayLike,
    s12: ArrayLike,
    s13: ArrayLike,
    s23: ArrayLike,
) -> NDArray[np.float64]:
    """
    Implementation from pylife

    Calculate equivalent stress according to von Mises.

    Parameters
    ----------
    s11: array_like
        Component 11 of 3D tensor.
    s22: array_like
        Component 22 of 3D tensor.
    s33: array_like
        Component 33 of 3D tensor.
    s12: array_like
        Component 12 of 3D tensor.
    s13: array_like
        Component 13 of 3D tensor.
    s23: array_like
        Component 23 of 3D tensor.

    Returns
    -------
    numpy.ndarray:
        Von Mises equivalent stress. Shape is the same as the components.

    Raises
    ------
    AssertionError
        Components' shape is not consistent.
    """
    s11 = np.asarray(s11, dtype=np.float64)
    s22 = np.asarray(s22, dtype=np.float64)
    s33 = np.asarray(s33, dtype=np.float64)
    s12 = np.asarray(s12, dtype=np.float64)
    s13 = np.asarray(s13, dtype=np.float64)
    s23 = np.asarray(s23, dtype=np.float64)

    assert (
        s11.shape == s22.shape
        and s11.shape == s33.shape
        and s11.shape == s12.shape
        and s11.shape == s13.shape
        and s11.shape == s23.shape
    ), "Components' shape is not consistent."

    mises_stress = np.sqrt(
        s11**2 + s22**2 + s33**2 - s11 * s22 - s11 * s33 - s22 * s33 + 3 * (s12**2 + s13**2 + s23**2)
    )

    return mises_stress
