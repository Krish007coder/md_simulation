"""
pbc.py — Periodic Boundary Conditions and Minimum Image Convention.

Functions:
    apply_pbc       — Wrap positions back into the box using modulo
    minimum_image   — Apply MIC to a displacement vector
"""

import numpy as np


def apply_pbc(positions: "np.ndarray", box) -> "np.ndarray":
    """
    Wrap particle positions into [0, box.L) using the modulo operator.

    Call this after every Verlet position update to keep particles inside.

    Args:
        positions: Shape (N, 3). Modified in-place.
        box      : Box object.

    Returns:
        positions: Wrapped positions.
    """
    positions[:] = positions % box.L
    return positions


def minimum_image(displacement: "np.ndarray", box) -> "np.ndarray":
    """
    Apply the Minimum Image Convention to a displacement vector.

    Adjusts each component of the displacement so that |d| <= L/2.
    This selects the nearest periodic image for distance calculations.

    Formula: d = d - L * round(d / L)

    Args:
        displacement: Shape (3,) or (N, 3). Displacement vector(s).
        box         : Box object.

    Returns:
        displacement: MIC-corrected displacement.
    """
    return displacement - box.L * np.round(displacement / box.L)