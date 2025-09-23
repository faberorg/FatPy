"""Test functions for stress calculations in structural mechanics.

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


@pytest.fixture
def stress_vector_2d() -> NDArray[np.float64]:
    """Fixture providing sample stress vectors in Voigt notation.

    Returns:
        NDArray[np.float64]: Sample stress vectors.
    """
    # Provide a collection of representative stress states (n,6):
    # 1) Uniaxial tension in x (sigma_xx = 100)
    # 2) Uniaxial compression in z (sigma_zz = -50)
    # 3) Pure hydrostatic (all normal = 30)
    # 4) Pure shear (sxy = 40)
    # 5) Mixed state
    arr = np.array(
        [
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -50.0, 0.0, 0.0, 0.0],
            [30.0, 30.0, 30.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 40.0],
            [np.sqrt(2.0), -np.sqrt(2), 0.0, 0.0, 0.0, np.sqrt(2.0)],
        ],
        dtype=np.float64,
    )

    return arr


@pytest.fixture
def principal_stresses_2d() -> NDArray[np.float64]:
    """Fixture providing expected principal stresses for the sample stress vectors.

    Returns:
        NDArray[np.float64]: Expected principal stresses.
    """
    arr = np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 0.0, -50.0],
            [30.0, 30.0, 30.0],
            [40.0, 0.0, -40.0],
            [2.0, 0.0, -2.0],
        ],
        dtype=np.float64,
    )

    return arr


# TODO: implement 3D cases
def stress_vector_3d(stress_vector_2d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fixture providing sample stress vectors in Voigt notation. 3D case.

    This is similar to the 2D case but with additional axis possibly representing time.

    Returns:
        NDArray[np.float64]: Sample stress vectors.
    """
    # Expand to 3D by adding an additional axis (e.g., time)
    return np.repeat(stress_vector_2d[np.newaxis, :, :], repeats=4, axis=0)


def test_calc_principal_stresses_and_directions_shape_2d(
    stress_vector_2d: NDArray[np.float64],
) -> None:
    """Test shape of principal stresses and directions output."""
    principals, directions = stress.calc_principal_stresses_and_directions(
        stress_vector_2d
    )
    assert principals.shape[0] == stress_vector_2d.shape[0]
    assert principals.shape[1] == 3

    assert directions.shape[0] == stress_vector_2d.shape[0]
    assert directions.shape[1] == 3
    assert directions.shape[2] == 3


def test_calc_principal_stresses_and_directions_ordering_2d(
    stress_vector_2d: NDArray[np.float64],
) -> None:
    """Test ordering of principal stresses (descending)."""
    principals, dirctions = stress.calc_principal_stresses_and_directions(
        stress_vector_2d
    )
    assert np.all(principals[:, 0] >= principals[:, 1])
    assert np.all(principals[:, 1] >= principals[:, 2])

    # test if directions order matches principal stresses order
    # by checking A*v = λ*v for each principal stress/direction pair
    for i in range(stress_vector_2d.shape[0]):
        sigma = stress_vector_2d[i]
        tensor = np.array(
            [
                [sigma[0], sigma[5], sigma[4]],
                [sigma[5], sigma[1], sigma[3]],
                [sigma[4], sigma[3], sigma[2]],
            ]
        )
        for j in range(3):
            eigvec = dirctions[i, :, j]
            eigval = principals[i, j]
            # Check if A*v = λ*v
            Av = tensor @ eigvec
            lv = eigval * eigvec
            assert np.allclose(Av, lv, atol=1e-12)


def test_calc_principal_stresses_and_directions_values_2d(
    stress_vector_2d: NDArray[np.float64],
    principal_stresses_2d: NDArray[np.float64],
) -> None:
    """Test specific known values of principal stresses."""
    principals, _ = stress.calc_principal_stresses_and_directions(stress_vector_2d)
    assert np.allclose(principals, principal_stresses_2d, atol=1e-12)


def test_calc_principal_stresses_shape_2d(
    stress_vector_2d: NDArray[np.float64],
) -> None:
    """Test shape of principal stresses output."""
    principals = stress.calc_principal_stresses(stress_vector_2d)
    assert principals.shape[0] == stress_vector_2d.shape[0]
    assert principals.shape[1] == 3


def test_calc_principal_stresses_ordering_2d(
    stress_vector_2d: NDArray[np.float64],
) -> None:
    """Test ordering of principal stresses (descending)."""
    principals = stress.calc_principal_stresses(stress_vector_2d)
    assert np.all(principals[:, 0] >= principals[:, 1])
    assert np.all(principals[:, 1] >= principals[:, 2])


def test_calc_principal_stresses_values_2d(
    stress_vector_2d: NDArray[np.float64],
    principal_stresses_2d: NDArray[np.float64],
) -> None:
    """Test specific known values of principal stresses."""
    principals = stress.calc_principal_stresses(stress_vector_2d)
    assert np.allclose(principals, principal_stresses_2d, atol=1e-12)


def test_calc_principal_directions_shape_2d(
    stress_vector_2d: NDArray[np.float64],
) -> None:
    """Test shape of principal directions output."""
    directions = stress.calc_principal_directions(stress_vector_2d)
    assert directions.shape[0] == stress_vector_2d.shape[0]
    assert directions.shape[1] == 3
    assert directions.shape[2] == 3


def test_calc_principal_directions_ordering_2d(
    stress_vector_2d: NDArray[np.float64],
) -> None:
    """Test ordering of principal directions matches principal stresses order."""
    principals = stress.calc_principal_stresses(stress_vector_2d)
    directions = stress.calc_principal_directions(stress_vector_2d)
    # For each stress state, check A*v = λ*v for each direction
    for i in range(stress_vector_2d.shape[0]):
        sigma = stress_vector_2d[i]
        tensor = np.array(
            [
                [sigma[0], sigma[5], sigma[4]],
                [sigma[5], sigma[1], sigma[3]],
                [sigma[4], sigma[3], sigma[2]],
            ]
        )
        for j in range(3):
            eigvec = directions[i, :, j]
            eigval = principals[i, j]
            Av = tensor @ eigvec
            lv = eigval * eigvec
            assert np.allclose(Av, lv, atol=1e-12)


def test_invariants_and_hydrostatic_deviator(
    stress_vector_2d: NDArray[np.float64],
) -> None:
    invariants = stress.calc_stress_invariants(stress_vector_2d)
    hydro = stress.calc_hydrostatic_stress(stress_vector_2d)
    deviator = stress.calc_stress_deviator(stress_vector_2d)
    # Shape checks
    assert invariants.shape == (stress_vector_2d.shape[0], 3)
    assert hydro.shape == (stress_vector_2d.shape[0],)
    assert deviator.shape == stress_vector_2d.shape
    # For hydrostatic case (index 2), deviator should be zero
    assert np.allclose(deviator[2], 0.0, atol=1e-12)
    # For hydrostatic case, invariants: I1 = 90, I2 = ? compute from definition
    I1 = invariants[2, 0]
    assert np.isclose(I1, 90.0)


def test_von_mises_and_signed_variants(stress_vector_2d: NDArray[np.float64]) -> None:
    vm = stress.calc_von_mises_stress(stress_vector_2d)
    svm_h = stress.calc_signed_von_mises_by_hydrostatic(stress_vector_2d)
    svm_p = stress.calc_signed_von_mises_by_max_abs_principal(stress_vector_2d)
    svm_i1 = stress.calc_signed_von_mises_by_first_invariant(stress_vector_2d)
    # Shape
    assert vm.shape == (stress_vector_2d.shape[0],)
    # Non-negative for von Mises
    assert np.all(vm >= 0.0)
    # For hydrostatic pure case index 2, von Mises == 0 and signed variants 0
    assert np.isclose(vm[2], 0.0)
    assert np.isclose(svm_h[2], 0.0)
    assert np.isclose(svm_p[2], 0.0)
    assert np.isclose(svm_i1[2], 0.0)
    # For pure shear case index 3 (sxy = 40): von Mises = sqrt(3/2)*|shear|*? check
    # Compute expected manually from definition used in implementation
    sx, sy, sz = stress_vector_2d[3, :3]
    sxy = stress_vector_2d[3, 5]
    expected_vm3 = np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2 + 6 * (sxy**2))
    )
    assert np.isclose(vm[3], expected_vm3)


def test_tresca_and_signed_variants(stress_vector_2d: NDArray[np.float64]) -> None:
    tresca = stress.calc_tresca_stress(stress_vector_2d)
    stresca_h = stress.calc_signed_tresca_by_hydrostatic(stress_vector_2d)
    stresca_p = stress.calc_signed_tresca_by_max_abs_principal(stress_vector_2d)
    # Shape
    assert tresca.shape == (stress_vector_2d.shape[0],)
    # Hydrostatic case (index 2) tresca == 0
    assert np.isclose(tresca[2], 0.0)
    assert np.isclose(stresca_h[2], 0.0)
    # For uniaxial tension (index 0), principal stresses should include 100 and two zeros -> tresca = (100 - 0)/2
    assert np.isclose(tresca[0], 50.0)
