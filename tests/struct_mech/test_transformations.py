"""Test functions for transformations in structural mechanics.

Tensor projection on a plane:

    1. In-plane basis: orthonormal, right-handed, l in the x-y plane, fallback
       l = x_hat when the normal is parallel to z_hat.
    2. Normal and shear components against closed forms (uniaxial stress on
       a rotated plane, pure shear, hydrostatic stress).
    3. Shapes: arbitrary leading dimensions are preserved.
    4. Invalid inputs raise ValueError.

"""

import numpy as np
import pytest
from numpy.typing import NDArray

from fatpy.struct_mech import transformations
from fatpy.struct_mech.transformations import plane_basis, tensor_projection_on_plane
from fatpy.utils import voigt


def unit(*components: float) -> NDArray[np.float64]:
    """Unit vector along the given components."""
    vector = np.array(components, dtype=np.float64)
    return vector / np.linalg.norm(vector)


NORMALS = [
    unit(1, 0, 0),
    unit(0, 1, 0),
    unit(0, 0, 1),
    unit(0, 0, -1),
    unit(1, 1, 0),
    unit(1, 2, 3),
    unit(-0.3, 0.8, -0.5),
    unit(1e-14, 0, 1),  # numerically parallel to z_hat
]


# -- in-plane basis -------------------------------------------------------------


@pytest.mark.parametrize("normal", NORMALS)
def test_plane_basis_is_right_handed_orthonormal(normal: NDArray[np.float64]) -> None:
    """(l, r, n) is orthonormal and l x r = n."""
    l_vec, r_vec = plane_basis(normal)
    frame = np.stack((l_vec, r_vec, normal))
    assert frame @ frame.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.cross(l_vec, r_vec) == pytest.approx(normal, abs=1e-12)


def test_plane_basis_definition() -> None:
    """l = normalise(z x n), r = n x l for a generic normal."""
    normal = unit(1, 2, 3)
    l_vec, r_vec = plane_basis(normal)
    expected_l = unit(-2, 1, 0)  # z x (1, 2, 3) = (-2, 1, 0)
    assert l_vec == pytest.approx(expected_l)
    assert r_vec == pytest.approx(np.cross(normal, expected_l))
    assert l_vec[2] == 0.0


@pytest.mark.parametrize(
    ("normal", "expected_r"),
    [(unit(0, 0, 1), [0.0, 1.0, 0.0]), (unit(0, 0, -1), [0.0, -1.0, 0.0])],
)
def test_plane_basis_fallback_for_normal_parallel_to_z(
    normal: NDArray[np.float64], expected_r: list[float]
) -> None:
    """n parallel to z_hat: l = x_hat, r = n x x_hat = +-y_hat."""
    l_vec, r_vec = plane_basis(normal)
    assert l_vec.tolist() == [1.0, 0.0, 0.0]
    assert r_vec == pytest.approx(expected_r)


# -- projection: closed forms ---------------------------------------------------


@pytest.mark.parametrize("angle_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 137.0])
def test_uniaxial_stress_on_rotated_plane(angle_deg: float) -> None:
    """sigma_xx on a plane at angle a: sigma_n = s cos^2 a, |tau| = s sin a cos a."""
    sigma = 120.0
    a = np.deg2rad(angle_deg)
    normal = np.array([np.cos(a), np.sin(a), 0.0])
    normal_component, shear = tensor_projection_on_plane(
        np.array([sigma, 0, 0, 0, 0, 0], dtype=np.float64), normal
    )
    assert normal_component == pytest.approx(sigma * np.cos(a) ** 2, abs=1e-12)
    assert np.linalg.norm(shear) == pytest.approx(
        abs(sigma * np.sin(a) * np.cos(a)), abs=1e-12
    )
    assert shear[1] == pytest.approx(0.0, abs=1e-12)  # no shear along r (= z here)


def test_pure_shear_on_plane_normal_to_x() -> None:
    """sigma_xy on the plane n = x_hat: no normal stress, shear tau along y."""
    tau = 75.0
    normal_component, shear = tensor_projection_on_plane(
        np.array([0, 0, 0, 0, 0, tau], dtype=np.float64), unit(1, 0, 0)
    )
    assert normal_component == pytest.approx(0.0, abs=1e-12)
    assert shear == pytest.approx([tau, 0.0])  # l = y_hat, r = z_hat


@pytest.mark.parametrize("normal", NORMALS)
def test_hydrostatic_stress_has_no_shear(normal: NDArray[np.float64]) -> None:
    """p I on any plane: sigma_n = p, tau = 0."""
    normal_component, shear = tensor_projection_on_plane(
        np.array([-40, -40, -40, 0, 0, 0], dtype=np.float64), normal
    )
    assert normal_component == pytest.approx(-40.0)
    assert shear == pytest.approx([0.0, 0.0], abs=1e-12)


@pytest.mark.parametrize("normal", NORMALS)
def test_traction_is_recovered(normal: NDArray[np.float64]) -> None:
    """sigma_n n + tau_l l + tau_r r equals the traction sigma n."""
    stress = np.array([80.0, -20.0, 35.0, 12.0, -7.0, 44.0])
    normal_component, shear = tensor_projection_on_plane(stress, normal)
    l_vec, r_vec = plane_basis(normal)
    rebuilt = normal_component * normal + shear[0] * l_vec + shear[1] * r_vec
    assert rebuilt == pytest.approx(voigt.voigt_to_tensor(stress) @ normal)


def test_strain_tensor_is_projected_the_same_way() -> None:
    """Tensor shear strain e_xy on n = x_hat gives the tensor strain e_xy."""
    normal_component, shear = tensor_projection_on_plane(
        np.array([0, 0, 0, 0, 0, 1.5e-4], dtype=np.float64), unit(1, 0, 0)
    )
    assert normal_component == pytest.approx(0.0, abs=1e-18)
    assert shear == pytest.approx([1.5e-4, 0.0])


# -- shapes -------------------------------------------------------------------------


@pytest.mark.parametrize("leading", [(), (7,), (4, 5), (1000, 200), (2, 3, 4)])
def test_leading_dimensions_are_preserved(leading: tuple[int, ...]) -> None:
    """(..., 6) gives (...) and (..., 2), elementwise."""
    rng = np.random.default_rng(0)
    stress = rng.normal(size=(*leading, 6))
    normal = unit(1, 2, 3)
    normal_component, shear = tensor_projection_on_plane(stress, normal)
    assert normal_component.shape == leading
    assert shear.shape == (*leading, 2)
    flat_n, flat_s = tensor_projection_on_plane(stress.reshape(-1, 6), normal)
    assert normal_component.reshape(-1) == pytest.approx(flat_n)
    assert shear.reshape(-1, 2) == pytest.approx(flat_s)


# -- invalid inputs -----------------------------------------------------------------


def test_wrong_voigt_shape_raises() -> None:
    """The last dimension must be 6."""
    with pytest.raises(ValueError, match="6 Voigt"):
        tensor_projection_on_plane(np.zeros((10, 3)), unit(1, 0, 0))


@pytest.mark.parametrize(
    ("normal", "message"),
    [
        (np.array([1.0, 0.0]), r"shape \(3,\)"),
        (np.array([[1.0, 0.0, 0.0]]), r"shape \(3,\)"),
        (np.array([np.nan, 0.0, 1.0]), "finite"),
        (np.array([1.0, 1.0, 0.0]), "unit vector"),
        (np.zeros(3), "unit vector"),
    ],
)
def test_invalid_normal_raises(normal: NDArray[np.float64], message: str) -> None:
    """The normal must be a finite unit vector of shape (3,)."""
    with pytest.raises(ValueError, match=message):
        tensor_projection_on_plane(np.zeros(6), normal)
    with pytest.raises(ValueError, match=message):
        plane_basis(normal)


def test_module_does_not_use_stress_module() -> None:
    """The projection depends on utils.voigt only, not on struct_mech.stress."""
    source = transformations.__file__
    with open(source, encoding="utf-8") as file:
        text = file.read()
    assert "struct_mech import stress" not in text
    assert "from fatpy.utils import voigt" in text
