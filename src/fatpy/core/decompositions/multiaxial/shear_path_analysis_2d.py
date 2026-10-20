r"""Shear Path Analysis 2D methods of multiaxial decompositions.

These methods are designed to evaluate the mean shear stress and shear stress amplitude
acting on a specified material plane, typically used in multiaxial fatigue analysis. By
projecting the stress tensor onto the plane of interest, they provide key parameters
for assessing crack initiation risks and fatigue life under complex loading paths.

Minimum circumscribed circle:
    The smallest circle enclosing a finite point set is unique and is
    supported either by **two** points forming a diameter or by **three**
    points lying on it (Papadopoulos 1994, 1998). Searching pairs only (the
    longest chord) misses the three-point configurations, such as acute or
    isosceles triangles, and underestimates the radius; see
    `longest_chord_circle`, kept only as a comparison baseline.

References:
    - I. V. Papadopoulos, "A new criterion of fatigue strength for
      out-of-phase bending and torsion of hard metals", Int. J. Fatigue 16
      (1994) 377-384.
    - I. V. Papadopoulos, "Critical plane approaches in high-cycle fatigue:
      on the definition of the amplitude and mean value of the shear stress
      acting on the critical plane", Fatigue Fract. Eng. Mater. Struct. 21
      (1998) 269-285.
    - E. Welzl, "Smallest enclosing disks (balls and ellipsoids)", in New
      Results and New Trends in Computer Science, LNCS 555 (1991) 359-370.
"""

import numpy as np
from numpy.typing import NDArray

#: Relative tolerance (to the size of the point set) of the containment test.
CONTAINMENT_TOLERANCE = 1e-12
#: Below this relative area, three points are taken as collinear.
COLLINEAR_TOLERANCE = 1e-12
# Fixed seed of the shuffle: the result is deterministic.
_SHUFFLE_SEED = 0


def _validate_path(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return `points` as float64 after checking the shape (n_steps, 2).

    Raises:
        ValueError: If `points` is not of shape (n_steps, 2) with n_steps >= 1,
            or contains NaN or inf.
    """
    path = np.asarray(points, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 1:
        raise ValueError(
            "The path must have shape exactly (n_steps, 2), got "
            f"{path.shape}. Multi-node or multi-plane arrays must be looped "
            "over by the caller, one (n_steps, 2) path at a time, so that the "
            "reduction is done over the time axis."
        )
    if not np.all(np.isfinite(path)):
        raise ValueError("The path must contain finite values only.")
    return path


def _circle_from_two(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Circle having the segment ab as a diameter."""
    return (a + b) / 2.0, float(np.linalg.norm(a - b)) / 2.0


def _circle_from_three(
    a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Circumscribed circle of the triangle abc.

    For three collinear points, no circle passes through all of them: the
    circle having the farthest pair as a diameter is returned.
    """
    ab, ac = b - a, c - a
    cross = ab[0] * ac[1] - ab[1] * ac[0]
    longest = max(float(ab @ ab), float(ac @ ac), float((c - b) @ (c - b)))
    if abs(cross) <= COLLINEAR_TOLERANCE * longest:
        pairs = ((a, b), (a, c), (b, c))
        return max(
            (_circle_from_two(p, q) for p, q in pairs), key=lambda circle: circle[1]
        )
    ab2, ac2 = float(ab @ ab), float(ac @ ac)
    offset = np.array([ac[1] * ab2 - ab[1] * ac2, ab[0] * ac2 - ac[0] * ab2]) / (
        2.0 * cross
    )
    return a + offset, float(np.linalg.norm(offset))


def _first_outside(
    points: NDArray[np.float64],
    centre: NDArray[np.float64],
    radius: float,
    tolerance: float,
    start: int,
    stop: int,
) -> int:
    """Index of the first point of points[start:stop] outside the circle, or -1."""
    if start >= stop:
        return -1
    distance = np.linalg.norm(points[start:stop] - centre, axis=1)
    outside = distance > radius + tolerance
    if not np.any(outside):
        return -1
    return start + int(np.argmax(outside))


def min_circumscribed_circle(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    r"""Smallest circle enclosing a path of points in a plane.

    Exact solution by the incremental algorithm of Welzl (1991), written
    without recursion: the points are visited in a fixed pseudo-random
    order; whenever a point lies outside the current circle, the circle is
    rebuilt with that point on its boundary, first as the smallest circle
    through it and one earlier point (**two-point** support, a diameter),
    then, if another earlier point still lies outside, as the circumscribed
    circle of three points (**three-point** support). Both support cases of
    the minimum circle are thus covered. The expected cost is linear in the
    number of distinct points. No optimiser is used.

    ??? abstract "Math Equations"
        The minimum circumscribed circle of the points
        $\mathbf{p}_1, \dots, \mathbf{p}_n$ solves

        $$
        (\mathbf{c}^*, R^*) = \arg\min_{\mathbf{c}}
        \max_{k} \lVert \mathbf{p}_k - \mathbf{c} \rVert ,
        $$

        with $R^*$ the shear stress amplitude and $\mathbf{c}^*$ the mean
        shear stress vector when the points are the shear stress path on a
        material plane (Papadopoulos 1998).

    Degenerate inputs: one point (or identical points) gives that point and
    radius 0; two points give their midpoint and half their distance; three
    collinear points give the circle having the farthest pair as diameter.
    Repeated points (e.g. several identical cycles) are removed first.

    Args:
        points: Path, shape exactly (n_steps, 2), n_steps >= 1.

    Returns:
        Tuple (centre, radius): centre of shape (2,) and radius (float), in
        the unit of `points`.

    Raises:
        ValueError: If `points` is not of shape (n_steps, 2), including any
            3D array, or contains NaN or inf.

    Example:
        >>> triangle = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 4.0]])
        >>> centre, radius = min_circumscribed_circle(triangle)
        >>> np.round(centre, 9).tolist(), round(radius, 9)
        ([3.0, 0.875], 3.125)
    """
    path = _validate_path(points)
    origin = path.mean(axis=0)  # work around the mean for accuracy
    unique = np.unique(path - origin, axis=0)
    if unique.shape[0] == 1:
        return unique[0] + origin, 0.0

    order = np.random.default_rng(_SHUFFLE_SEED).permutation(unique.shape[0])
    p = unique[order]
    n = p.shape[0]
    tolerance = CONTAINMENT_TOLERANCE * float(np.max(np.abs(p)))

    centre, radius = p[0].copy(), 0.0
    i = _first_outside(p, centre, radius, tolerance, 1, n)
    while i >= 0:
        centre, radius = p[i].copy(), 0.0  # p[i] lies on the boundary
        j = _first_outside(p, centre, radius, tolerance, 0, i)
        while j >= 0:
            centre, radius = _circle_from_two(p[i], p[j])  # p[i], p[j] on it
            k = _first_outside(p, centre, radius, tolerance, 0, j)
            while k >= 0:
                centre, radius = _circle_from_three(p[i], p[j], p[k])
                k = _first_outside(p, centre, radius, tolerance, k + 1, j)
            j = _first_outside(p, centre, radius, tolerance, j + 1, i)
        i = _first_outside(p, centre, radius, tolerance, i + 1, n)
    return centre + origin, radius


def longest_chord_circle(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    r"""Circle on the longest chord of a path: NON-CONSERVATIVE baseline.

    !!! warning "Comparison baseline only"
        This is **not** a registered component and must **not** be used in a
        fatigue criterion. It searches pairs of points only and returns the
        circle having the farthest pair as diameter. It misses the
        three-point support of the minimum circumscribed circle and so
        **underestimates** the shear stress amplitude whenever the minimum
        circle is supported by three points (e.g. acute or isosceles
        triangles). It is kept to evidence that point; use
        `min_circumscribed_circle`.

    ??? abstract "Math Equations"
        $$
        R_{chord} = \tfrac{1}{2} \max_{j,k} \lVert \mathbf{p}_j -
        \mathbf{p}_k \rVert \le R^* \le \tfrac{2}{\sqrt{3}} R_{chord}
        $$

        (Jung's theorem): the underestimate reaches 13.4 % for an
        equilateral triangle. The returned circle may not even contain all
        the points.

    Args:
        points: Path, shape exactly (n_steps, 2), n_steps >= 1.

    Returns:
        Tuple (centre, radius): midpoint of the longest chord, shape (2,),
        and half its length (float).

    Raises:
        ValueError: If `points` is not of shape (n_steps, 2), including any
            3D array, or contains NaN or inf.

    Example:
        >>> triangle = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 4.0]])
        >>> centre, radius = longest_chord_circle(triangle)
        >>> centre.tolist(), radius
        ([3.0, 0.0], 3.0)
    """
    path = _validate_path(points)
    unique = np.unique(path, axis=0)
    best = (0, 0, 0.0)
    for i in range(unique.shape[0] - 1):  # O(n^2) time, O(n) memory
        distance = np.sum((unique[i + 1 :] - unique[i]) ** 2, axis=1)
        k = int(np.argmax(distance))
        if distance[k] > best[2]:
            best = (i, i + 1 + k, float(distance[k]))
    return _circle_from_two(unique[best[0]], unique[best[1]])
