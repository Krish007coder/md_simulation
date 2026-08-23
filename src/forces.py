"""
forces.py — Lennard-Jones force and potential energy.

The LJ pair potential:
    U(r) = 4ε [ (σ/r)^12 − (σ/r)^6 ]

The LJ force on particle i due to particle j:
    F_ij = (48ε/r²) [ (σ/r)^12 − 0.5(σ/r)^6 ] × r_vec_ij

In reduced units (σ = ε = 1):
    F_ij = (48/r²) [ r^{-12} − 0.5 r^{-6} ] × r_vec_ij

Functions:
    lj_potential          — U(r) for scalar r
    lj_force_scalar       — |F(r)| for scalar r
    compute_forces_and_pe — All forces + total PE for the system
"""

import numpy as np


def lj_potential(r: float, sigma: float = 1.0, epsilon: float = 1.0) -> float:
    """
    Lennard-Jones potential: U(r) = 4ε [(σ/r)^12 − (σ/r)^6].

    Args:
        r      : Scalar distance > 0.
        sigma  : LJ length parameter.
        epsilon: LJ energy parameter.

    Returns:
        float: Potential energy U(r).
    """
    sr  = sigma / r
    sr6 = sr ** 6
    return 4.0 * epsilon * (sr6 ** 2 - sr6)


def lj_force_scalar(r: float, sigma: float = 1.0, epsilon: float = 1.0) -> float:
    """
    LJ force magnitude: F(r) = -dU/dr = (4ε/r)[12(σ/r)^12 − 6(σ/r)^6].

    Positive = repulsive, negative = attractive.
    """
    sr  = sigma / r
    sr6 = sr ** 6
    return (4.0 * epsilon / r) * (12.0 * sr6 ** 2 - 6.0 * sr6)


def compute_forces_and_pe(
    positions: "np.ndarray",
    box,
    sigma: float = 1.0,
    epsilon: float = 1.0
) -> "tuple[np.ndarray, float]":
    """
    Compute LJ forces on all particles and total potential energy.

    Uses double loop with i < j to avoid double-counting.
    Applies Minimum Image Convention to pairwise displacements.
    Uses Newton's Third Law: F_ji = −F_ij.

    Args:
        positions: Shape (N, 3). Particle positions.
        box      : Box object (for MIC via box.L).
        sigma    : LJ σ parameter.
        epsilon  : LJ ε parameter.

    Returns:
        forces (np.ndarray): Shape (N, 3). Net force on each particle.
        pe     (float)     : Total potential energy.
    """
    N      = len(positions)
    forces = np.zeros((N, 3), dtype=np.float64)
    pe     = 0.0

    for i in range(N - 1):
        for j in range(i + 1, N):

            # Displacement vector (with Minimum Image Convention)
            r_ij  = positions[j] - positions[i]
            r_ij  = r_ij - box.L * np.round(r_ij / box.L)

            # Squared distance and its inverse
            r_sq  = np.dot(r_ij, r_ij)
            inv_r2 = 1.0 / r_sq

            # LJ quantities
            sr6   = (sigma ** 2 * inv_r2) ** 3   # (σ/r)^6
            sr12  = sr6 ** 2                       # (σ/r)^12

            # Pair potential energy
            pe += 4.0 * epsilon * (sr12 - sr6)

            # Force prefactor: df = (48*sr12 - 24*sr6) / r^2
            fac    = (48.0 * sr12 - 24.0 * sr6) * inv_r2
            f_ij   = fac * r_ij                    # force vector

            # Newton's 3rd Law: apply equal-and-opposite forces
            forces[i] += f_ij
            forces[j] -= f_ij

    return forces, pe