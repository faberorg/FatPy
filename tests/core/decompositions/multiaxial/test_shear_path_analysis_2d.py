"""Test functions for the 2D shear path analysis (circle geometry).

Minimum circumscribed circle:

    1. Reference paths reproduced exactly: proportional (with and without an
       offset), circle, general ellipse (major semi-axis).
    2. Three-point support: an acute isosceles triangle, where no pair of
       points is a diameter, against its hand-computed circumradius; the
       longest-chord baseline is strictly smaller there.
    3. Degenerate inputs: one point, two points, collinear points, repeated
       points.
    4. Cross-check against an independent brute-force enumeration of all
       pairs and all triads (written only in this file).
    5. Wrong shapes raise ValueError.

"""

import ast
import itertools
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from fatpy.core.decompositions.multiaxial import shear_path_analysis_2d
from fatpy.core.decompositions.multiaxial.shear_path_analysis_2d import (
    longest_chord_circle,
    min_circumscribed_circle,
)

REL = 1e-12


def brute_force_circle(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Smallest enclosing circle by enumerating every pair and every triad.

    Independent of the implementation: every pair gives the circle on its
    diameter, every non-collinear triad its circumcircle (from a 2x2 linear
    system); the smallest candidate containing all points is returned.
    """
    if len(points) == 1:
        return points[0].copy(), 0.0
    scale = float(np.max(np.abs(points))) or 1.0
    candidates: list[tuple[NDArray[np.float64], float]] = []
    for a, b in itertools.combinations(points, 2):
        candidates.append(((a + b) / 2, float(np.linalg.norm(a - b)) / 2))
    for a, b, c in itertools.combinations(points, 3):
        matrix = 2 * np.array([b - a, c - a])
        if abs(np.linalg.det(matrix)) < 1e-12 * scale**2:
            continue
        rhs = np.array([b @ b - a @ a, c @ c - a @ a])
        centre = np.linalg.solve(matrix, rhs)
        candidates.append((centre, float(np.linalg.norm(a - centre))))
    enclosing = [
        (centre, radius)
        for centre, radius in candidates
        if np.all(np.linalg.norm(points - centre, axis=1) <= radius + 1e-9 * scale)
    ]
    return min(enclosing, key=lambda circle: circle[1])


def sample_times(n_steps: int, extra: tuple[float, ...] = ()) -> NDArray[np.float64]:
    """n_steps phases over one cycle, plus extra phases [rad]."""
    return np.concatenate((np.linspace(0, 2 * np.pi, n_steps, endpoint=False), extra))


# -- reference paths --------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"), [(100.0, 0.0), (0.0, 50.0), (3.0, 4.0), (-7.0, 2.5)]
)
def test_proportional_path(a: float, b: float) -> None:
    """C(t) = [A sin wt, B sin wt]: radius sqrt(A^2 + B^2), centre at origin."""
    t = sample_times(40)  # contains pi/2 and 3 pi/2
    path = np.column_stack((a * np.sin(t), b * np.sin(t)))
    centre, radius = min_circumscribed_circle(path)
    assert radius == pytest.approx(np.hypot(a, b), rel=REL)
    assert centre == pytest.approx([0.0, 0.0], abs=1e-12 * np.hypot(a, b))


def test_proportional_path_with_offset() -> None:
    """A constant offset moves the centre only."""
    t = sample_times(40)
    offset = np.array([250.0, -80.0])
    path = np.column_stack((3.0 * np.sin(t), 4.0 * np.sin(t))) + offset
    centre, radius = min_circumscribed_circle(path)
    assert radius == pytest.approx(5.0, rel=1e-11)
    assert centre == pytest.approx(offset, rel=1e-12)


@pytest.mark.parametrize("n_steps", [3, 4, 7, 72, 1000])
def test_circular_path(n_steps: int) -> None:
    """Points on a circle of radius R: radius R, centre at its centre."""
    t = sample_times(n_steps) + 0.3
    centre_true = np.array([12.0, -5.0])
    path = centre_true + 40.0 * np.column_stack((np.cos(t), np.sin(t)))
    centre, radius = min_circumscribed_circle(path)
    assert radius == pytest.approx(40.0, rel=REL)
    assert centre == pytest.approx(centre_true, rel=REL)


@pytest.mark.parametrize(
    ("f", "g", "p", "q"),
    [
        (3.0, 1.0, 0.5, 2.0),
        (100.0, 0.0, 0.0, 57.7),
        (1.0, 2.0, -3.0, 0.5),
        (2, 2, 1, 1),
    ],
)
def test_elliptical_path(f: float, g: float, p: float, q: float) -> None:
    """C(t) = [f cos + g sin, p cos + q sin]: radius = major semi-axis.

    The phase of the major vertex is added to the samples so that the
    sampled path reaches the ellipse's extreme points exactly.
    """
    s = f**2 + g**2 + p**2 + q**2
    major = np.sqrt(s / 2 + np.sqrt((s / 2) ** 2 - (f * q - g * p) ** 2))
    _, _, vt = np.linalg.svd(np.array([[f, g], [p, q]]))
    vertex = float(np.arctan2(vt[0, 1], vt[0, 0]))
    t = sample_times(37, (vertex, vertex + np.pi))
    path = np.column_stack(
        (f * np.cos(t) + g * np.sin(t), p * np.cos(t) + q * np.sin(t))
    )
    centre, radius = min_circumscribed_circle(path)
    assert radius == pytest.approx(major, rel=1e-11)
    assert centre == pytest.approx([0.0, 0.0], abs=1e-10 * major)


# -- three-point support and the longest-chord baseline -----------------------------


def test_acute_isosceles_triangle() -> None:
    """(0,0), (6,0), (3,4): no pair is a diameter.

    Hand computation: sides a = 6, b = c = 5, area K = 12,
    R = abc / (4K) = 150 / 48 = 3.125, centre (3, 4 - 3.125) = (3, 0.875).
    The longest chord is 6, giving 3.0 < 3.125, and its circle does not
    contain the apex (distance 4 from (3, 0)).
    """
    triangle = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 4.0]])
    centre, radius = min_circumscribed_circle(triangle)
    assert radius == pytest.approx(3.125, rel=REL)
    assert centre == pytest.approx([3.0, 0.875], rel=REL)

    chord_centre, chord_radius = longest_chord_circle(triangle)
    assert chord_radius == pytest.approx(3.0, rel=REL)
    assert chord_radius < radius
    assert np.linalg.norm(triangle[2] - chord_centre) > chord_radius


def test_equilateral_triangle_reaches_the_jung_bound() -> None:
    """Side s: R = s / sqrt(3), longest chord s / 2, ratio 2 / sqrt(3)."""
    side = 10.0
    t = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    triangle = side / np.sqrt(3) * np.column_stack((np.cos(t), np.sin(t)))
    _, radius = min_circumscribed_circle(triangle)
    _, chord_radius = longest_chord_circle(triangle)
    assert radius == pytest.approx(side / np.sqrt(3), rel=REL)
    assert radius / chord_radius == pytest.approx(2 / np.sqrt(3), rel=REL)


def test_obtuse_triangle_has_two_point_support() -> None:
    """Obtuse triangle: the longest side is the diameter, both methods agree."""
    triangle = np.array([[0.0, 0.0], [10.0, 0.0], [4.0, 1.0]])
    centre, radius = min_circumscribed_circle(triangle)
    chord_centre, chord_radius = longest_chord_circle(triangle)
    assert radius == pytest.approx(5.0, rel=REL)
    assert centre == pytest.approx([5.0, 0.0], rel=REL)
    assert chord_radius == pytest.approx(radius, rel=REL)
    assert chord_centre == pytest.approx(centre, rel=REL)


# -- degenerate inputs --------------------------------------------------------------


def test_single_point() -> None:
    """One point: centre at the point, radius 0."""
    centre, radius = min_circumscribed_circle(np.array([[2.5, -1.0]]))
    assert centre.tolist() == [2.5, -1.0]
    assert radius == 0.0
    assert longest_chord_circle(np.array([[2.5, -1.0]]))[1] == 0.0


def test_identical_points() -> None:
    """Repeated identical points: radius 0."""
    centre, radius = min_circumscribed_circle(np.tile([[7.0, 3.0]], (50, 1)))
    assert centre == pytest.approx([7.0, 3.0])
    assert radius == 0.0


def test_two_points() -> None:
    """Two points: midpoint and half the distance."""
    centre, radius = min_circumscribed_circle(np.array([[1.0, 1.0], [4.0, 5.0]]))
    assert centre == pytest.approx([2.5, 3.0])
    assert radius == pytest.approx(2.5)


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_three_collinear_points(order: tuple[int, int, int]) -> None:
    """Collinear points in any order: the farthest pair is the diameter."""
    points = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 6.0]])[list(order)]
    centre, radius = min_circumscribed_circle(points)
    assert centre == pytest.approx([1.5, 3.0])
    assert radius == pytest.approx(np.hypot(3.0, 6.0) / 2)


def test_many_collinear_points() -> None:
    """Points on a line (a proportional path) with repetitions."""
    s = np.sin(np.linspace(0, 6 * np.pi, 301))
    path = np.column_stack((2.0 * s, -1.0 * s))
    centre, radius = min_circumscribed_circle(path)
    assert radius == pytest.approx(np.hypot(2.0, 1.0), rel=1e-11)
    assert centre == pytest.approx([0.0, 0.0], abs=1e-12)


# -- brute-force cross-check and invariances ---------------------------------------


@pytest.mark.parametrize("seed", range(60))
def test_matches_brute_force_on_random_point_sets(seed: int) -> None:
    """Random sets of 2 to 12 points agree with the exhaustive enumeration."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 13))
    points = rng.normal(scale=rng.uniform(0.1, 100), size=(n, 2))
    if seed % 3 == 0:  # integer grid: ties, cocircular and collinear points
        points = rng.integers(-3, 4, size=(n, 2)).astype(np.float64)
    centre, radius = min_circumscribed_circle(points)
    centre_bf, radius_bf = brute_force_circle(np.unique(points, axis=0))
    assert radius == pytest.approx(radius_bf, rel=1e-9, abs=1e-12)
    assert centre == pytest.approx(centre_bf, rel=1e-9, abs=1e-9)
    assert np.all(np.linalg.norm(points - centre, axis=1) <= radius * (1 + 1e-9))


def test_longest_chord_never_exceeds_the_true_radius() -> None:
    """R_chord <= R* <= 2 / sqrt(3) R_chord on random sets."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        points = rng.normal(size=(int(rng.integers(3, 30)), 2))
        _, radius = min_circumscribed_circle(points)
        _, chord = longest_chord_circle(points)
        assert chord <= radius * (1 + 1e-12)
        assert radius <= 2 / np.sqrt(3) * chord * (1 + 1e-12)


def test_rigid_motion_and_order_invariance() -> None:
    """Rotating, translating or reordering the points moves the circle rigidly."""
    rng = np.random.default_rng(5)
    points = rng.normal(size=(200, 2))
    centre, radius = min_circumscribed_circle(points)
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    shift = np.array([30.0, -12.0])
    moved = points[rng.permutation(200)] @ rotation.T + shift
    centre_m, radius_m = min_circumscribed_circle(moved)
    assert radius_m == pytest.approx(radius, rel=1e-12)
    assert centre_m == pytest.approx(rotation @ centre + shift, rel=1e-12)


def test_large_path() -> None:
    """A long path (10 000 steps) is solved exactly."""
    t = np.linspace(0, 2 * np.pi, 10_000, endpoint=False)
    path = np.column_stack((80 * np.cos(t), 80 * np.sin(t))) + [5.0, 5.0]
    centre, radius = min_circumscribed_circle(path)
    assert radius == pytest.approx(80.0, rel=REL)
    assert centre == pytest.approx([5.0, 5.0], rel=REL)


# -- wrong shapes and values ----------------------------------------------------------


@pytest.mark.parametrize(
    "points",
    [
        np.zeros(2),
        np.zeros((5, 3)),
        np.zeros((0, 2)),
        np.zeros((1000, 200, 2)),  # nodes x steps: ambiguous, refused
        np.zeros((3, 5, 2)),
        np.zeros((2, 5)),
    ],
)
def test_wrong_shape_raises(points: NDArray[np.float64]) -> None:
    """Anything but (n_steps, 2) raises ValueError, 3D included."""
    with pytest.raises(ValueError, match=r"exactly \(n_steps, 2\)"):
        min_circumscribed_circle(points)
    with pytest.raises(ValueError, match=r"exactly \(n_steps, 2\)"):
        longest_chord_circle(points)


def test_non_finite_values_raise() -> None:
    """NaN or inf in the path raises ValueError."""
    with pytest.raises(ValueError, match="finite"):
        min_circumscribed_circle(np.array([[0.0, 1.0], [np.nan, 2.0]]))


def test_module_imports_nothing_from_struct_mech() -> None:
    """The solver stays free of stress-specific imports (numpy only)."""
    tree = ast.parse(Path(shear_path_analysis_2d.__file__).read_text(encoding="utf-8"))
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert sorted(set(imported)) == ["numpy", "numpy.typing"]
