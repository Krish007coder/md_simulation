"""
energy.py — Energy computation.

Functions:
    kinetic_energy — KE = (1/2) * m * Σ|v_i|²
    temperature    — T = 2*KE / (3*N)    [reduced units, k_B = 1]
    total_energy   — E = KE + PE
"""

import numpy as np


def kinetic_energy(velocities: "np.ndarray", mass: float = 1.0) -> float:
    """
    Total kinetic energy: KE = (1/2) * m * Σ_i |v_i|²

    Args:
        velocities: Shape (N, 3).
        mass      : Particle mass (default 1.0).

    Returns:
        float: Total kinetic energy.
    """
    return 0.5 * mass * np.sum(velocities ** 2)


def temperature(velocities: "np.ndarray") -> float:
    """
    Instantaneous temperature: T = 2*KE / (3*N)

    From the equipartition theorem (k_B = 1 in reduced units).

    Args:
        velocities: Shape (N, 3).

    Returns:
        float: Temperature in reduced units.
    """
    N  = len(velocities)
    ke = kinetic_energy(velocities)
    return 2.0 * ke / (3.0 * N)


def total_energy(ke: float, pe: float) -> float:
    """
    Total mechanical energy: E = KE + PE

    Args:
        ke: Kinetic energy.
        pe: Potential energy.

    Returns:
        float: Total energy.
    """
    return ke + pe