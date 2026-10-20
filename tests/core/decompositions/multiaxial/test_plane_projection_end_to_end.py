"""End-to-end tests: tensor projection on a plane chained with the circle.

The caller chains the two components: the stress history of one node,
shape (n_steps, 6), is projected on one plane, giving the shear path of
shape (n_steps, 2), whose minimum circumscribed circle gives the shear
stress amplitude (radius) and mean (centre). Several planes and nodes are
looped over explicitly, one (n_steps, 2) path at a time.

    1. Fully reversed uniaxial tension, planes at 45 degrees to the axis:
       shear amplitude sigma_a / 2, normal stress sigma / 2.
    2. Fully reversed torsion, plane normal to the axis: shear amplitude
       tau_a, normal stress 0.
    3. Proportional loading: on a principal plane the shear vanishes; on any
       other plane the shear path is a straight segment through the origin.
    4. Rotating the in-plane basis (or flipping the normal) leaves the radius
       unchanged and rotates the centre.

"""

import numpy as np
import pytest
from numpy.typing import NDArray

from fatpy.core.decompositions.multiaxial.shear_path_analysis_2d import (
    min_circumscribed_circle,
)
from fatpy.struct_mech.transformations import plane_basis, tensor_projection_on_plane
from fatpy.utils import voigt

N_STEPS = 72  # multiple of 4: the samples contain the peaks of sin and cos
T = np.linspace(0.0, 2.0 * np.pi, N_STEPS, endpoint=False)


def unit(*components: float) -> NDArray[np.float64]:
    """Unit vector along the given components."""
    vector = np.array(components, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def voigt_history(**components: NDArray[np.float64]) -> NDArray[np.float64]:
    """Voigt stress history (n_steps, 6) from named components s11 ... s12."""
    history = np.zeros((N_STEPS, 6))
    for name, values in components.items():
        history[:, ("s11", "s22", "s33", "s23", "s13", "s12").index(name)] = values
    return history


def amplitude_on_plane(
    history: NDArray[np.float64], normal: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Chain both components: (normal stress, shear path, centre, radius)."""
    normal_stress, shear_path = tensor_projection_on_plane(history, normal)
    centre, radius = min_circumscribed_circle(shear_path)
    return normal_stress, shear_path, centre, radius


def test_uniaxial_tension_on_planes_at_45_degrees() -> None:
    """sigma_11 = sigma_a sin wt: shear amplitude sigma_a / 2 on every 45 deg plane."""
    sigma_a = 200.0
    history = voigt_history(s11=sigma_a * np.sin(T))
    planes = [unit(1, 1, 0), unit(1, -1, 0), unit(1, 0, 1), unit(1, 0, -1)]
    planes += [unit(1, np.cos(b), np.sin(b)) for b in (0.3, 1.1, 2.0)]
    for normal in planes:  # explicit loop: one (n_steps, 2) path per plane
        normal_stress, _, centre, radius = amplitude_on_plane(history, normal)
        assert radius == pytest.approx(sigma_a / 2, rel=1e-12)
        assert centre == pytest.approx([0.0, 0.0], abs=1e-12)
        assert normal_stress == pytest.approx(sigma_a / 2 * np.sin(T), abs=1e-12)


def test_uniaxial_tension_plane_angle_dependence() -> None:
    """Amplitude sigma_a |sin a cos a|, normal stress sigma cos^2 a, any angle a."""
    sigma_a = 150.0
    history = voigt_history(s11=sigma_a * np.sin(T))
    for angle in np.deg2rad([0.0, 10.0, 30.0, 45.0, 70.0, 90.0]):
        normal = np.array([np.cos(angle), np.sin(angle), 0.0])
        normal_stress, _, _, radius = amplitude_on_plane(history, normal)
        assert radius == pytest.approx(
            sigma_a * abs(np.sin(angle) * np.cos(angle)), abs=1e-12
        )
        assert normal_stress == pytest.approx(
            sigma_a * np.cos(angle) ** 2 * np.sin(T), abs=1e-12
        )


def test_torsion_on_plane_normal_to_the_axis() -> None:
    """sigma_12 = tau_a sin wt, shaft axis x: amplitude tau_a, no normal stress."""
    tau_a = 115.0
    history = voigt_history(s12=tau_a * np.sin(T))
    normal_stress, _, centre, radius = amplitude_on_plane(history, unit(1, 0, 0))
    assert radius == pytest.approx(tau_a, rel=1e-12)
    assert centre == pytest.approx([0.0, 0.0], abs=1e-12)
    assert normal_stress == pytest.approx(np.zeros(N_STEPS), abs=1e-12)


def test_several_nodes_are_looped_explicitly() -> None:
    """(n_nodes, n_steps, 6) is projected at once, the circle node by node."""
    scales = np.array([1.0, 2.0, 0.5, 3.0])
    histories = scales[:, None, None] * voigt_history(s12=100.0 * np.sin(T))
    normal_stress, shear = tensor_projection_on_plane(histories, unit(1, 0, 0))
    assert shear.shape == (4, N_STEPS, 2)
    assert normal_stress.shape == (4, N_STEPS)
    radii = [min_circumscribed_circle(shear[node])[1] for node in range(4)]
    assert radii == pytest.approx(100.0 * scales, rel=1e-12)


def proportional_history() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """sigma(t) = sin(wt) sigma_0 with a general sigma_0 (Voigt)."""
    sigma_0 = np.array([120.0, -40.0, 30.0, 25.0, -15.0, 60.0])
    return np.sin(T)[:, None] * sigma_0, sigma_0


def test_principal_plane_has_no_shear() -> None:
    """On a plane normal to a principal direction the shear path is the origin."""
    history, sigma_0 = proportional_history()
    principal, directions = np.linalg.eigh(voigt.voigt_to_tensor(sigma_0))
    for value, normal in zip(principal, directions.T, strict=True):
        normal_stress, shear_path, centre, radius = amplitude_on_plane(history, normal)
        assert np.abs(shear_path).max() == pytest.approx(0.0, abs=1e-10)
        assert radius == pytest.approx(0.0, abs=1e-10)
        assert normal_stress == pytest.approx(value * np.sin(T), abs=1e-10)


def test_proportional_shear_path_is_a_segment_through_the_origin() -> None:
    """Any other plane: collinear shear vectors, centre 0, radius |tau_0|."""
    history, sigma_0 = proportional_history()
    for normal in (unit(1, 2, 3), unit(-1, 0.5, 2), unit(0, 1, 1)):
        _, shear_0 = tensor_projection_on_plane(sigma_0, normal)
        normal_stress, shear_path, centre, radius = amplitude_on_plane(history, normal)
        cross = shear_path[:, 0] * shear_0[1] - shear_path[:, 1] * shear_0[0]
        assert np.abs(cross).max() == pytest.approx(0.0, abs=1e-10)
        assert centre == pytest.approx([0.0, 0.0], abs=1e-10)
        assert radius == pytest.approx(float(np.linalg.norm(shear_0)), rel=1e-12)
        expected_normal = normal @ voigt.voigt_to_tensor(sigma_0) @ normal
        assert normal_stress == pytest.approx(expected_normal * np.sin(T), abs=1e-10)


def test_rotating_the_in_plane_basis_leaves_the_radius_unchanged() -> None:
    """Non-proportional loading: rotated or reflected basis, same radius."""
    history = voigt_history(
        s11=200.0 * np.sin(T), s12=115.0 * np.cos(T), s33=40.0 * np.sin(2 * T) + 30.0
    )
    normal = unit(1, 2, 2)
    _, shear_path, centre, radius = amplitude_on_plane(history, normal)
    assert radius > 0.0
    for angle in (0.4, 1.3, np.pi / 2, 3.0):
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        for q in (rotation, rotation @ np.diag([1.0, -1.0])):  # rotation, reflection
            centre_q, radius_q = min_circumscribed_circle(shear_path @ q.T)
            assert radius_q == pytest.approx(radius, rel=1e-12)
            assert centre_q == pytest.approx(q @ centre, abs=1e-10)

    # Flipping the normal changes the basis but not the radius or sigma_n.
    normal_stress, _, _, _ = amplitude_on_plane(history, normal)
    normal_stress_flip, _, _, radius_flip = amplitude_on_plane(history, -normal)
    assert radius_flip == pytest.approx(radius, rel=1e-12)
    assert normal_stress_flip == pytest.approx(normal_stress, rel=1e-12)

    # An explicit other basis (l rotated about n) gives the same radius.
    l_vec, r_vec = plane_basis(normal)
    other_l = np.cos(0.8) * l_vec + np.sin(0.8) * r_vec
    other_r = np.cross(normal, other_l)
    traction = voigt.voigt_to_tensor(history) @ normal  # (n_steps, 3)
    other_path = np.column_stack((traction @ other_l, traction @ other_r))
    assert min_circumscribed_circle(other_path)[1] == pytest.approx(radius, rel=1e-12)
