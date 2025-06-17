"""The module contains methods evaluating mean shear stress and amplitude on a plane."""


def min_circumscribed_circle() -> None:
    """The method is part of decompositions methods.

    Method implements a geometric algorithm to find the smallest circle enclosing
    a set of points representing a stress path on a plane.

    Outputs radius and center of the circle.
    """
    raise NotImplementedError


def min_circumscribed_ellipse() -> None:
    """The method is part of decompositions methods.

    Method implements a geometric algorithm to find the smallest ellipse enclosing
    a set of points representing a stress path on a plane.

    Outputs length of semi axes and center of the ellipse.
    """
    raise NotImplementedError


def max_prismatic_hull() -> None:
    """The method is part of decompositions methods.

    This method finds maximum prismatic hull enclosing stress path
    in 5D Ilyushin's space.

    Outputs a measure of the shear stress stress amplitude and the mean shear stress.
    """
    raise NotImplementedError


def moment_of_inertia_method() -> None:
    """The method is part of decompositions methods.

    This method evaluates the shear stress amplitude and mean using
    the momen tof inertia approach on a plane.
    Please refer to papers by Meggiolaro and Castro for more details.
    """
    raise NotImplementedError
