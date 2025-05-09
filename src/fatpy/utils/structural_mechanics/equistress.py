"""Module implements functions for calculating equivalent stress."""

import numpy as np
from numpy.typing import NDArray


def calc_von_mises_stress[T: NDArray[np.floating]](voigt_stress_vec: T) -> T:
    """Calculates equivalent stress according to von Mises.

    Parameters
    ----------
    voigt_stress_vec : numpy.ndarray
        Stress tensor of shape [N, 6], where N is the number of points.
        Expected format of voigt_stress_vec: [s11, s22, s33, s23, s13, s12].

    Returns:
    -------
    numpy.ndarray
        Von Mises equivalent stress of shape [N].

    Raises:
    ------
        None

    """
    str_vec = voigt_stress_vec

    mises_stress = np.sqrt(
        str_vec[:, 0] ** 2
        + str_vec[:, 1] ** 2
        + str_vec[:, 2] ** 2
        - str_vec[:, 0] * str_vec[:, 1]
        - str_vec[:, 0] * str_vec[:, 2]
        - str_vec[:, 1] * str_vec[:, 2]
        + 3 * (str_vec[:, 3] ** 2 + str_vec[:, 4] ** 2 + str_vec[:, 5] ** 2)
    )

    return mises_stress  # type: ignore
