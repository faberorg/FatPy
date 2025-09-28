"""Test functions for stress calculations in structural mechanics.

Voigt notation:
    Voigt notation tests are not repeated here as they are covered in related tests.

Eigenvalues and eigenvectors:

    Only features related to eigenvalue/vector modification from `np.linalg.eig` are
    tested as testing of eigenvalue/vector calculation itself is out of scope.
    Following features are tested:

        1. Shape of outputs
        2. Ordering of principal stresses (descending)

"""

import numpy as np
import pytest
from numpy.typing import NDArray

from fatpy.struct_mech import stress
from fatpy.utils import voigt


@pytest.fixture
def stress_vector_sample() -> NDArray[np.float64]:
    """Fixture providing sample stress vectors in Voigt notation (3D: shape (2, 3, 6)).

    Returns:
        NDArray[np.float64]: Sample stress vectors.
    """
    # Two sets, each with three representative stress states (3,6):
    arr = np.array(
        [
            [
                [100, 0, 0, 0, 0, 0],  # Uniaxial tension in x
                [0, 0, -50, 0, 0, 0],  # Uniaxial compression in z
                [30, 30, 30, 0, 0, 0],  # Pure hydrostatic
            ],
            [
                [0, 0, 0, 0, 0, 40],  # Pure shear
                [np.sqrt(2), -np.sqrt(2), 0, 0, 0, np.sqrt(2)],  # Mixed state
                [14, 0, 6, 0, 3, 0],  # Another mixed state
            ],
        ],
        dtype=np.float64,
    )

    return arr


@pytest.fixture
def principal_stresses_sample() -> NDArray[np.float64]:
    """Fixture providing expected principal stresses for the sample stress vectors.

    Returns:
        NDArray[np.float64]: Expected principal stresses.
    """
    arr = np.array(
        [
            [
                [100, 0, 0],
                [0, 0, -50],
                [30, 30, 30],
            ],
            [
                [40, 0, -40],
                [2, 0, -2],
                [15, 5, 0],
            ],
        ],
        dtype=np.float64,
    )

    return arr


@pytest.fixture
def hydrostatic_stress_sample() -> NDArray[np.float64]:
    """Fixture providing expected hydrostatic stresses for the sample stress vectors.

    Returns:
        NDArray[np.float64]: Expected hydrostatic stresses.
    """
    arr = np.array(
        [
            [100 / 3, -50 / 3, 30],
            [0, 0, 20 / 3],
        ],
        dtype=np.float64,
    )

    return arr


def test_calc_principal_stresses_and_directions_shape(
    stress_vector_sample: NDArray[np.float64],
) -> None:
    """Test shape of principal stresses and directions output."""
    principals, directions = stress.calc_principal_stresses_and_directions(
        stress_vector_sample
    )
    assert principals.shape[:-1] == stress_vector_sample.shape[:-1]
    assert principals.shape[-1] == 3


def test_calc_principal_stresses_and_directions_ordering(
    stress_vector_sample: NDArray[np.float64],
) -> None:
    """Test ordering of principal stresses (descending)."""
    principals, directions = stress.calc_principal_stresses_and_directions(
        stress_vector_sample
    )
    assert np.all(principals[..., 0] >= principals[..., 1])
    assert np.all(principals[..., 1] >= principals[..., 2])

    # test if directions order matches principal stresses order
    # by checking A*v = λ*v for each principal stress/direction pair
    for idx in np.ndindex(principals.shape[:-1]):
        stress_tensor = voigt.voigt_to_tensor(stress_vector_sample[idx])
        for i in range(3):
            principal_stress = principals[idx + (i,)]
            direction = directions[idx + (slice(None), i)]
            Av = stress_tensor @ direction
            lv = principal_stress * direction
            assert np.allclose(Av, lv, atol=1e-12)


def test_calc_principal_stresses_and_directions(
    stress_vector_sample: NDArray[np.float64],
    principal_stresses_sample: NDArray[np.float64],
) -> None:
    """Test shape of principal stresses and directions output."""
    principals, directions = stress.calc_principal_stresses_and_directions(
        stress_vector_sample
    )

    assert np.allclose(principals, principal_stresses_sample, atol=1e-12)


def test_calc_principal_stresses(
    stress_vector_sample: NDArray[np.float64],
    principal_stresses_sample: NDArray[np.float64],
) -> None:
    """Test shape of principal stresses output."""
    principals = stress.calc_principal_stresses(stress_vector_sample)

    assert principals.shape[:-1] == stress_vector_sample.shape[:-1]
    assert principals.shape[-1] == 3

    assert np.all(principals[..., 0] >= principals[..., 1])
    assert np.all(principals[..., 1] >= principals[..., 2])

    assert np.allclose(principals, principal_stresses_sample, atol=1e-12)


def test_calc_stress_invariants(
    stress_vector_sample: NDArray[np.float64],
) -> None:
    invariants = stress.calc_stress_invariants(stress_vector_sample)

    assert invariants.shape == stress_vector_sample.shape[:-1] + (3,)

    for idx in np.ndindex(invariants.shape[:-1]):
        stress_tensor = voigt.voigt_to_tensor(stress_vector_sample[idx])

        I1 = np.trace(stress_tensor)
        I2 = 0.5 * (I1**2 - np.trace(stress_tensor @ stress_tensor))
        I3 = np.linalg.det(stress_tensor)

        assert np.isclose(invariants[idx + (0,)], I1, atol=1e-12)
        assert np.isclose(invariants[idx + (1,)], I2, atol=1e-12)
        assert np.isclose(invariants[idx + (2,)], I3, atol=1e-12)


def test_calc_hydrostatic_stress(
    stress_vector_sample: NDArray[np.float64],
    hydrostatic_stress_sample: NDArray[np.float64],
) -> None:
    hydrostatic = stress.calc_hydrostatic_stress(stress_vector_sample)

    assert hydrostatic.shape == stress_vector_sample.shape[:-1]

    assert np.allclose(hydrostatic, hydrostatic_stress_sample, atol=1e-12)


def test_calc_stress_deviator(
    stress_vector_sample: NDArray[np.float64],
) -> None:
    deviator = stress.calc_stress_deviator(stress_vector_sample)

    assert deviator.shape == stress_vector_sample.shape

    for idx in np.ndindex(deviator.shape[:-1]):
        stress_tensor = voigt.voigt_to_tensor(stress_vector_sample[idx])
        hydrostatic = np.trace(stress_tensor) / 3.0
        hydrostatic_tensor = np.eye(3) * hydrostatic
        deviator_tensor = stress_tensor - hydrostatic_tensor
        deviator_voigt = voigt.tensor_to_voigt(deviator_tensor)
        assert np.allclose(deviator[idx], deviator_voigt, atol=1e-12)


def test_calc_von_mises(
    stress_vector_sample: NDArray[np.float64],
) -> None:
    """Test calculation of von Mises stress.

    Uses the definition based on the stress deviator tensor.

    Args:
        stress_vector_sample (NDArray[np.float64]): Sample stress vectors.
    """
    von_mises_stress = stress.calc_von_mises_stress(stress_vector_sample)

    assert von_mises_stress.shape == stress_vector_sample.shape[:-1]

    for idx in np.ndindex(von_mises_stress.shape):
        stress_tensor = voigt.voigt_to_tensor(stress_vector_sample[idx])
        s = stress_tensor - np.eye(3) * (np.trace(stress_tensor) / 3.0)
        vm = np.sqrt(1.5 * np.sum(s**2))
        assert np.isclose(von_mises_stress[idx], vm, atol=1e-12)


def test_calc_signed_von_mises_by_hydrostatic(
    stress_vector_sample: NDArray[np.float64],
) -> None:
    """Test calculation of signed von Mises stress.

    Uses the definition based on the hydrostatic stress.

    Args:
        stress_vector_sample (NDArray[np.float64]): Sample stress vectors.
    """
    signed_von_mises = stress.calc_signed_von_mises_by_hydrostatic(stress_vector_sample)

    assert signed_von_mises.shape == stress_vector_sample.shape[:-1]

    for idx in np.ndindex(signed_von_mises.shape):
        stress_tensor = voigt.voigt_to_tensor(stress_vector_sample[idx])
        s = stress_tensor - np.eye(3) * (np.trace(stress_tensor) / 3.0)
        vm = np.sqrt(1.5 * np.sum(s**2))
        hydrostatic = np.trace(stress_tensor) / 3.0
        sign = np.sign(hydrostatic)
        sign = 1.0 if np.isclose(hydrostatic, 0.0) else sign
        assert np.isclose(signed_von_mises[idx], sign * vm, atol=1e-12)


def test_calc_signed_von_mises_by_max_abs_principal(
    stress_vector_sample: NDArray[np.float64],
) -> None:
    """Test calculation of signed von Mises stress.

    Uses the definition based on the principal stress with the largest absolute value.

    Args:
        stress_vector_sample (NDArray[np.float64]): Sample stress vectors.
    """
    signed_von_mises = stress.calc_signed_von_mises_by_max_abs_principal(
        stress_vector_sample
    )

    assert signed_von_mises.shape == stress_vector_sample.shape[:-1]

    for idx in np.ndindex(signed_von_mises.shape):
        stress_tensor = voigt.voigt_to_tensor(stress_vector_sample[idx])
        s = stress_tensor - np.eye(3) * (np.trace(stress_tensor) / 3.0)
        vm = np.sqrt(1.5 * np.sum(s**2))
        principals = np.linalg.eigvalsh(stress_tensor)
        avg13 = 0.5 * (principals[0] + principals[2])
        sign = np.sign(avg13)
        sign = 1.0 if np.isclose(avg13, 0.0) else sign
        svm = sign * vm
        assert np.isclose(signed_von_mises[idx], svm, atol=1e-12)


def test_calc_signed_von_mises_by_first_invariant(
    stress_vector_sample: NDArray[np.float64],
) -> None:
    """Test calculation of signed von Mises stress.

    Uses the definition based on the first stress invariant.

    Args:
        stress_vector_sample (NDArray[np.float64]): Sample stress vectors.
    """
    signed_von_mises = stress.calc_signed_von_mises_by_first_invariant(
        stress_vector_sample
    )

    assert signed_von_mises.shape == stress_vector_sample.shape[:-1]

    for idx in np.ndindex(signed_von_mises.shape):
        stress_tensor = voigt.voigt_to_tensor(stress_vector_sample[idx])
        s = stress_tensor - np.eye(3) * (np.trace(stress_tensor) / 3.0)
        vm = np.sqrt(1.5 * np.sum(s**2))
        I1 = np.trace(stress_tensor)
        sign = np.sign(I1)
        sign = 1.0 if np.isclose(I1, 0.0) else sign
        svm = sign * vm
        assert np.isclose(signed_von_mises[idx], svm, atol=1e-12)


def test_calc_tresca(
    stress_vector_sample: NDArray[np.float64],
) -> None:
    """Test calculation of Tresca stress.

    Args:
        stress_vector_sample (NDArray[np.float64]): Sample stress vectors.
    """
    tresca_stress = stress.calc_tresca_stress(stress_vector_sample)

    assert tresca_stress.shape == stress_vector_sample.shape[:-1]

    for idx in np.ndindex(tresca_stress.shape):
        stress_tensor = voigt.voigt_to_tensor(stress_vector_sample[idx])
        principals = np.linalg.eigvalsh(stress_tensor)
        tresca = (principals[2] - principals[0]) / 2
        assert np.isclose(tresca_stress[idx], tresca, atol=1e-12)


def test_calc_signed_tresca_by_hydrostatic(
    stress_vector_sample: NDArray[np.float64],
) -> None:
    """Test calculation of signed Tresca stress.

    Uses the definition based on the hydrostatic stress.

    Args:
        stress_vector_sample (NDArray[np.float64]): Sample stress vectors.
    """
    signed_tresca = stress.calc_signed_tresca_by_hydrostatic(stress_vector_sample)

    assert signed_tresca.shape == stress_vector_sample.shape[:-1]

    for idx in np.ndindex(signed_tresca.shape):
        stress_tensor = voigt.voigt_to_tensor(stress_vector_sample[idx])
        principals = np.linalg.eigvalsh(stress_tensor)
        tresca = (principals[2] - principals[0]) / 2
        hydrostatic = np.trace(stress_tensor) / 3.0
        sign = np.sign(hydrostatic)
        sign = 1.0 if np.isclose(hydrostatic, 0.0) else sign
        assert np.isclose(signed_tresca[idx], sign * tresca, atol=1e-12)


def test_calc_signed_tresca_by_max_abs_principal(
    stress_vector_sample: NDArray[np.float64],
) -> None:
    """Test calculation of signed Tresca stress.

    Uses the definition based on the principal stress with the largest absolute value.

    Args:
        stress_vector_sample (NDArray[np.float64]): Sample stress vectors.
    """
    signed_tresca = stress.calc_signed_tresca_by_max_abs_principal(stress_vector_sample)

    assert signed_tresca.shape == stress_vector_sample.shape[:-1]

    for idx in np.ndindex(signed_tresca.shape):
        stress_tensor = voigt.voigt_to_tensor(stress_vector_sample[idx])
        principals = np.linalg.eigvalsh(stress_tensor)
        tresca = (principals[2] - principals[0]) / 2
        avg13 = 0.5 * (principals[0] + principals[2])
        sign = np.sign(avg13)
        sign = 1.0 if np.isclose(avg13, 0.0) else sign
        assert np.isclose(signed_tresca[idx], sign * tresca, atol=1e-12)
