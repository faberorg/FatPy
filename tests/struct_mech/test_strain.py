"""Test functions for strain calculations in structural mechanics.

Voigt notation:
    Voigt notation tests are not repeated here as they are covered in related tests.

Eigenvalues and eigenvectors:

    Only features related to eigenvalue/vector modification from `np.linalg.eig` are
    tested as testing of eigenvalue/vector calculation itself is out of scope.
    Following features are tested:

        1. Shape of outputs
        2. Ordering of principal strains (descending)

"""

import numpy as np
import pytest
from numpy.typing import NDArray

from fatpy.struct_mech import strain
from fatpy.utils import voigt


@pytest.fixture
def strain_vector_sample() -> NDArray[np.float64]:
    """Fixture providing sample strain vectors in Voigt notation (3D: shape (2, 3, 6)).

    Returns:
        NDArray[np.float64]: Sample strain vectors.
    """
    # Two sets, each with three representative strain states (3,6):
    arr = np.array(
        [
            [
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],  # Uniaxial strain in x
                [0.0, 0.0, -0.005, 0.0, 0.0, 0.0],  # Uniaxial compression in z
                [0.02, 0.02, 0.02, 0.0, 0.0, 0.0],  # Pure volumetric
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.03],  # Pure shear
                [0.008, 0.008, 0.0, 0.0, 0.0, 0.006],  # Mixed state
                [0.014, 0.0, 0.006, 0.0, 0.003, 0.0],  # Another mixed
            ],
        ],
        dtype=np.float64,
    )

    return arr


@pytest.fixture
def principal_strains_sample() -> NDArray[np.float64]:
    """Fixture providing expected principal strains for the sample strain vectors.

    Returns:
        NDArray[np.float64]: Expected principal strains.
    """
    arr = np.array(
        [
            [
                [0.01, 0.0, 0.0],
                [0.0, 0.0, -0.005],
                [0.02, 0.02, 0.02],
            ],
            [
                [0.03, 0.0, -0.03],
                [0.014, 0.002, 0.0],
                [0.015, 0.005, 0.0],
            ],
        ],
        dtype=np.float64,
    )

    return arr


@pytest.fixture
def volumetric_strain_sample() -> NDArray[np.float64]:
    """Fixture providing expected volumetric strains for the sample strain vectors.

    Returns:
        NDArray[np.float64]: Expected volumetric strains.
    """
    arr = np.array(
        [
            [(0.01 + 0.0 + 0.0) / 3.0, (0.0 + 0.0 + -0.005) / 3.0, 0.02],
            [0.0, (0.008 + 0.008 + 0.0) / 3.0, (0.014 + 0.0 + 0.006) / 3.0],
        ],
        dtype=np.float64,
    )

    return arr


def test_calc_principal_strains_and_directions_shape(
    strain_vector_sample: NDArray[np.float64],
) -> None:
    """Test shape of principal strains and directions output."""
    principals, directions = strain.calc_principal_strains_and_directions(
        strain_vector_sample
    )
    assert principals.shape[:-1] == strain_vector_sample.shape[:-1]
    assert principals.shape[-1] == 3

    assert directions.shape[:-2] == strain_vector_sample.shape[:-1]
    assert directions.shape[-2:] == (3, 3)  # 3x3 eigenvector matrix


def test_calc_principal_strains_and_directions_ordering(
    strain_vector_sample: NDArray[np.float64],
) -> None:
    """Test ordering of principal strains (descending)."""
    principals, directions = strain.calc_principal_strains_and_directions(
        strain_vector_sample
    )
    assert np.all(principals[..., 0] >= principals[..., 1])
    assert np.all(principals[..., 1] >= principals[..., 2])

    # test if directions order matches principal strains order
    # by checking A*v = λ*v for each principal strain/direction pair
    for idx in np.ndindex(principals.shape[:-1]):
        strain_tensor = voigt.voigt_to_tensor(strain_vector_sample[idx])
        for i in range(3):
            principal_strain = principals[idx + (i,)]
            direction = directions[idx + (slice(None), i)]
            Av = strain_tensor @ direction
            lv = principal_strain * direction
            assert np.allclose(Av, lv, atol=1e-12)


def test_calc_principal_strains(
    strain_vector_sample: NDArray[np.float64],
    principal_strains_sample: NDArray[np.float64],
) -> None:
    principals = strain.calc_principal_strains(strain_vector_sample)

    assert principals.shape[:-1] == strain_vector_sample.shape[:-1]
    assert principals.shape[-1] == 3

    assert np.all(principals[..., 0] >= principals[..., 1])
    assert np.all(principals[..., 1] >= principals[..., 2])

    assert np.allclose(principals, principal_strains_sample, atol=1e-12)


def test_calc_strain_invariants(
    strain_vector_sample: NDArray[np.float64],
) -> None:
    invariants = strain.calc_strain_invariants(strain_vector_sample)

    assert invariants.shape == strain_vector_sample.shape[:-1] + (3,)

    for idx in np.ndindex(invariants.shape[:-1]):
        strain_tensor = voigt.voigt_to_tensor(strain_vector_sample[idx])

        I1 = np.trace(strain_tensor)
        I2 = 0.5 * (I1**2 - np.trace(strain_tensor @ strain_tensor))
        I3 = np.linalg.det(strain_tensor)

        assert np.isclose(invariants[idx + (0,)], I1, atol=1e-12)
        assert np.isclose(invariants[idx + (1,)], I2, atol=1e-12)
        assert np.isclose(invariants[idx + (2,)], I3, atol=1e-12)


def test_calc_volumetric_strain(
    strain_vector_sample: NDArray[np.float64],
    volumetric_strain_sample: NDArray[np.float64],
) -> None:
    volumetric = strain.calc_volumetric_strain(strain_vector_sample)

    assert volumetric.shape == strain_vector_sample.shape[:-1]

    assert np.allclose(volumetric, volumetric_strain_sample, atol=1e-12)


def test_calc_deviatoric_strain(
    strain_vector_sample: NDArray[np.float64],
) -> None:
    deviator = strain.calc_deviatoric_strain(strain_vector_sample)

    assert deviator.shape == strain_vector_sample.shape

    for idx in np.ndindex(deviator.shape[:-1]):
        strain_tensor = voigt.voigt_to_tensor(strain_vector_sample[idx])
        hydro = np.trace(strain_tensor) / 3.0
        hydro_tensor = np.eye(3) * hydro
        deviator_tensor = strain_tensor - hydro_tensor
        deviator_voigt = voigt.tensor_to_voigt(deviator_tensor)
        assert np.allclose(deviator[idx], deviator_voigt, atol=1e-12)


def test_calc_von_mises_from_principals(
    strain_vector_sample: NDArray[np.float64],
) -> None:
    """Test calculation of von Mises strain from principals.

    Uses the definition based on the strain deviator tensor.

    Args:
        strain_vector_sample (NDArray[np.float64]): Sample strain vectors.
    """
    von_mises = strain.calc_von_mises_strain_from_principals(strain_vector_sample)

    assert von_mises.shape == strain_vector_sample.shape[:-1]

    for idx in np.ndindex(von_mises.shape):
        strain_tensor = voigt.voigt_to_tensor(strain_vector_sample[idx])
        s = strain_tensor - np.eye(3) * (np.trace(strain_tensor) / 3.0)
        vm = np.sqrt((2.0 / 3.0) * np.sum(s**2))
        assert np.isclose(von_mises[idx], vm, atol=1e-12)


def test_calc_von_mises_voigt(
    strain_vector_sample: NDArray[np.float64],
) -> None:
    """Test calculation of von Mises strain from Voigt components.

    Uses the principals method for verification.

    Args:
        strain_vector_sample (NDArray[np.float64]): Sample strain vectors.
    """
    von_mises = strain.calc_von_mises_strain_voigt(strain_vector_sample)

    assert von_mises.shape == strain_vector_sample.shape[:-1]

    strain_tensor = voigt.voigt_to_tensor(strain_vector_sample)
    principals = np.linalg.eigvalsh(strain_tensor)
    e1 = principals[..., 0]
    e2 = principals[..., 1]
    e3 = principals[..., 2]
    vm = np.sqrt((2.0 / 9.0) * ((e1 - e2) ** 2 + (e2 - e3) ** 2 + (e3 - e1) ** 2))
    assert np.allclose(von_mises, vm, atol=1e-12)


def test_calc_signed_von_mises_by_max_abs_principal(
    strain_vector_sample: NDArray[np.float64],
) -> None:
    signed_von_mises = strain.calc_signed_von_mises_by_max_abs_principal(
        strain_vector_sample
    )

    assert signed_von_mises.shape == strain_vector_sample.shape[:-1]

    for idx in np.ndindex(signed_von_mises.shape):
        strain_tensor = voigt.voigt_to_tensor(strain_vector_sample[idx])
        s = strain_tensor - np.eye(3) * (np.trace(strain_tensor) / 3.0)
        vm = np.sqrt((2.0 / 3.0) * np.sum(s**2))
        principals = np.linalg.eigvalsh(strain_tensor)
        indices = np.argmax(np.abs(principals))
        max_abs = principals[indices]
        sign = np.sign(max_abs)
        sign = np.where(np.isclose(max_abs, 0.0), 1.0, sign)
        assert np.isclose(signed_von_mises[idx], sign * vm, atol=1e-12)
