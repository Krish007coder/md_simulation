"""
box.py — Simulation box geometry.
"""

import numpy as np


class Box:
    """
    Cubic simulation box with side length L.

    Attributes:
        L      (float): Side length in reduced units.
        volume (float): Volume = L³.
        half_L (float): L/2, used for Minimum Image Convention.
    """

    def __init__(self, L: float):
        if L <= 0:
            raise ValueError(f"Box length must be positive, got {L}")
        self.L      = float(L)
        self.volume = L ** 3
        self.half_L = L / 2.0

    def __repr__(self) -> str:
        return f"Box(L={self.L}, volume={self.volume:.2f}, half_L={self.half_L})"
