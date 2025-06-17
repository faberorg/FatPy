"""This module contains methods for plane-based methods.

Methods for processing stress tensor path on a material plane.
Provides basic infrastructure for prediction methods based on
critical-plane and integral approaches.
"""


def define_local_coordinate_system() -> None:
    """Defines a local Cartesian coordinate system with the z-direction.

    The z-direction is aligned with a local surface normal line.
    """
    raise NotImplementedError


def describe_plane_orientation() -> None:
    """Defines the spatial orientation of a plane.

    The definition is provided in the local coordinate system
    using Euler angles.
    """
    raise NotImplementedError("This method is not implemented yet.")


def optimize_damage_parameter() -> None:
    """Optimizes the damage parameter.

    This method searches for a maximum (of any damage parameter), starting
    from a maximum value obtained from a previous scan using a regular
    pattern of Euler angles.
    """
    raise NotImplementedError("This method is not implemented yet.")


def critical_plane_integral_switch() -> None:
    """Handles the value of the damage parameter on the analyzed plane.

    Decides whether to check if this value is extreme over all planes
    (critical plane methods) or to integrate it with all previous results
    (integral methods).
    """
    raise NotImplementedError("This method is not implemented yet.")
