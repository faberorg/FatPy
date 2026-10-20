r"""Transformations methods of structural mechanics.

These routines perform mathematical operations to transform stress and strain tensors
between different coordinate systems or reference frames. They include tensor rotation,
transformation to principal axes, and projection onto specified planes, enabling
detailed analysis of material response under complex loading conditions and supporting
criteria based on critical plane, invariant, or principal stress/strain approaches.
Includes 5D deviatoric stress reduction.

Conventions:
    - Tensors use Voigt notation with shape (..., 6), as in `fatpy.utils.voigt`:

            (σ_11, σ_22, σ_33, σ_23, σ_13, σ_12)
            (σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy)

      Leading dimensions are preserved: the routines are elementwise, so an
      array of shape (n_nodes, n_steps, 6) gives results of shape
      (n_nodes, n_steps) for scalars and (n_nodes, n_steps, 2) for in-plane
      vectors.
    - Strain tensors are projected with the same routine; their shear
      components must be tensor strains (ε_12 = γ_12 / 2) for the result to
      be the tensor shear strain on the plane.
    - A plane is given by its unit normal vector $\mathbf{n}$, shape (3,),
      in the same coordinate system as the tensor.
    - In-plane vectors are given in the basis $(\mathbf{l}, \mathbf{r})$ of
      `plane_basis`: $\mathbf{l} = \operatorname{normalise}(\hat{\mathbf{z}}
      \times \mathbf{n})$, $\mathbf{r} = \mathbf{n} \times \mathbf{l}$.
"""

import numpy as np
from numpy.typing import NDArray

from fatpy.utils import voigt

#: Tolerance on the norm of a plane normal vector.
NORMAL_TOLERANCE = 1e-9
#: Below this length of z_hat x n, n is taken as parallel to z_hat.
PARALLEL_TOLERANCE = 1e-12


def _check_normal(normal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return `normal` as a float64 array after checking it is a unit vector.

    Raises:
        ValueError: If `normal` is not of shape (3,), not finite, or not of
            unit length within `NORMAL_TOLERANCE`.
    """
    n = np.asarray(normal, dtype=np.float64)
    if n.shape != (3,):
        raise ValueError(f"Plane normal must have shape (3,), got {n.shape}.")
    if not np.all(np.isfinite(n)):
        raise ValueError(f"Plane normal must be finite, got {n}.")
    length = float(np.linalg.norm(n))
    if abs(length - 1.0) > NORMAL_TOLERANCE:
        raise ValueError(f"Plane normal must be a unit vector, got norm {length!r}.")
    return n


def plane_basis(
    normal: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""In-plane orthonormal basis $(\mathbf{l}, \mathbf{r})$ of a plane.

    ??? abstract "Math Equations"
        $$
        \mathbf{l} = \frac{\hat{\mathbf{z}} \times \mathbf{n}}
                          {\lVert \hat{\mathbf{z}} \times \mathbf{n} \rVert},
        \qquad
        \mathbf{r} = \mathbf{n} \times \mathbf{l}
        $$

    The basis is built directly from $\mathbf{n}$, without spherical angles,
    and reproduces the $\mathbf{l}$ and $\mathbf{r}$ vectors of Cizova and of
    the Pisa scripts. $(\mathbf{l}, \mathbf{r}, \mathbf{n})$ is a right-handed
    orthonormal frame ($\mathbf{l} \times \mathbf{r} = \mathbf{n}$), and
    $\mathbf{l}$ lies in the global x-y plane.

    Degenerate case: if $\mathbf{n}$ is parallel to $\hat{\mathbf{z}}$
    ($\lVert \hat{\mathbf{z}} \times \mathbf{n} \rVert$ <
    `PARALLEL_TOLERANCE`), the cross product vanishes and
    $\mathbf{l} = \hat{\mathbf{x}}$ is used instead, so that
    $\mathbf{r} = \pm\hat{\mathbf{y}}$ for $\mathbf{n} = \pm\hat{\mathbf{z}}$.

    Args:
        normal: Unit normal of the plane, shape (3,).

    Returns:
        Tuple (l, r) of unit vectors, each of shape (3,).

    Raises:
        ValueError: If `normal` is not a finite unit vector of shape (3,).

    Example:
        >>> l, r = plane_basis(np.array([1.0, 0.0, 0.0]))
        >>> l.tolist(), r.tolist()
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    """
    n = _check_normal(normal)
    cross = np.cross(np.array([0.0, 0.0, 1.0]), n)
    length = float(np.linalg.norm(cross))
    if length < PARALLEL_TOLERANCE:
        l_vec = np.array([1.0, 0.0, 0.0])
    else:
        l_vec = cross / length
    r_vec = np.cross(n, l_vec)
    return l_vec, r_vec


def tensor_projection_on_plane(
    vector: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Project a stress or strain tensor onto a plane.

    The traction vector on the plane of unit normal $\mathbf{n}$ is split into
    its normal component and its two in-plane (shear) components along the
    basis $(\mathbf{l}, \mathbf{r})$ of `plane_basis`.

    ??? abstract "Math Equations"
        $$
        \mathbf{t} = \boldsymbol{\sigma}\,\mathbf{n}, \qquad
        \sigma_n = \mathbf{n} \cdot \mathbf{t}, \qquad
        \boldsymbol{\tau} = (\tau_l, \tau_r)
        = (\mathbf{l} \cdot \mathbf{t},\; \mathbf{r} \cdot \mathbf{t})
        $$

        with $\mathbf{l} = \operatorname{normalise}(\hat{\mathbf{z}} \times
        \mathbf{n})$ and $\mathbf{r} = \mathbf{n} \times \mathbf{l}$
        ($\mathbf{l} = \hat{\mathbf{x}}$ if $\mathbf{n} \parallel
        \hat{\mathbf{z}}$).

    ??? note "Choice of the in-plane basis"
        Any other orthonormal in-plane basis differs from $(\mathbf{l},
        \mathbf{r})$ by a rotation (or a reflection) about $\mathbf{n}$, so
        the shear coordinates $(\tau_l, \tau_r)$ of every time step are
        transformed by the same 2x2 orthogonal matrix $\mathbf{Q}$. Such a
        map preserves all distances between points, hence it maps the
        minimum circumscribed circle of the shear path onto the minimum
        circumscribed circle of the transformed path: the **radius** (the
        shear stress amplitude) is invariant, while the **centre** is
        transformed as $\mathbf{c}' = \mathbf{Q}\,\mathbf{c}$, so its
        components depend on the basis. Only the centre's length (the mean
        shear stress magnitude) is basis independent.

    Args:
        vector: Stress or strain tensor in Voigt notation, shape (..., 6).
            Leading dimensions are preserved.
        normal: Unit normal of the plane, shape (3,).

    Returns:
        Tuple (normal_component, shear):
            - normal_component: Normal stress (strain) on the plane, shape
              (...).
            - shear: In-plane shear components $(\tau_l, \tau_r)$, shape
              (..., 2).

    Raises:
        ValueError: If the last dimension of `vector` is not 6, or if
            `normal` is not a finite unit vector of shape (3,).

    Example:
        >>> sigma = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # tension in x
        >>> n = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)  # plane at 45 deg
        >>> normal_component, shear = tensor_projection_on_plane(sigma, n)
        >>> round(float(normal_component), 9), np.round(shear, 9).tolist()
        (50.0, [-50.0, 0.0])
    """
    voigt.check_shape(vector)
    l_vec, r_vec = plane_basis(normal)
    n = np.asarray(normal, dtype=np.float64)

    tensor = voigt.voigt_to_tensor(np.asarray(vector, dtype=np.float64))
    traction = tensor @ n  # (..., 3)
    normal_component = traction @ n
    shear = np.stack((traction @ l_vec, traction @ r_vec), axis=-1)
    return normal_component, shear
