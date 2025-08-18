"""Contains the function to calculate equivalent stresses."""

import numpy as np
from numpy.typing import NDArray


def calc_von_mises_stress[T: NDArray[np.floating]](stress_vector: T) -> T:
    """Calculates equivalent stress according to von Mises.

    Parameters
    ----------
    stress_vector : numpy.ndarray
        Voight stress tensor of shape [N, 6], where N is the number of points.
        Expected format of stress_vector: [s11, s22, s33, s23, s13, s12].

    Returns:
    -------
    numpy.ndarray
        Von Mises equivalent stress of shape [N].

    Raises:
    ------
        None

    """

    mises_stress = np.sqrt(
        stress_vector[:, 0] ** 2
        + stress_vector[:, 1] ** 2
        + stress_vector[:, 2] ** 2
        - stress_vector[:, 0] * stress_vector[:, 1]
        - stress_vector[:, 0] * stress_vector[:, 2]
        - stress_vector[:, 1] * stress_vector[:, 2]
        + 3 * (stress_vector[:, 3] ** 2
        + stress_vector[:, 4] ** 2 + stress_vector[:, 5] ** 2)
    )

    return mises_stress  # type: ignore
