"""
integrator.py — Verlet time integration.

Verlet algorithm:
    r(t+Δt) = 2r(t) − r(t−Δt) + a(t)Δt²
    where a = F/m = F  (m = 1 in reduced units)

Velocity estimation (central difference):
    v(t) = [r(t+Δt) − r(t−Δt)] / (2Δt)

Functions:
    bootstrap_prev_positions — Create ghost r(−Δt) for step 0
    verlet_step              — Advance positions by one timestep
    estimate_velocities      — Compute velocities from positions
    velocity_verlet_step_1   — Velocity Verlet half-step 1
    velocity_verlet_step_2   — Velocity Verlet half-step 2
"""

import numpy as np


def velocity_verlet_step_1(
    positions: "np.ndarray",
    velocities: "np.ndarray",
    forces: "np.ndarray",
    dt: float,
    box
) -> "tuple[np.ndarray, np.ndarray]":
    """
    First half of Velocity Verlet algorithm.

    v(t + Δt/2) = v(t) + (1/2) F(t) Δt
    r(t + Δt)   = r(t) + v(t + Δt/2) Δt
    """
    vel_half = velocities + 0.5 * forces * dt
    pos_new  = (positions + vel_half * dt) % box.L
    return pos_new, vel_half


def velocity_verlet_step_2(
    vel_half: "np.ndarray",
    forces_new: "np.ndarray",
    dt: float
) -> "np.ndarray":
    """
    Second half of Velocity Verlet algorithm.

    v(t + Δt) = v(t + Δt/2) + (1/2) F(t + Δt) Δt
    """
    return vel_half + 0.5 * forces_new * dt


def bootstrap_prev_positions(
    positions: "np.ndarray",
    velocities: "np.ndarray",
    dt: float
) -> "np.ndarray":
    """
    Estimate r(−Δt) for bootstrapping the Verlet algorithm.

    At t=0 we have r(0) and v(0) but not r(−Δt).
    Approximation: r(−Δt) ≈ r(0) − v(0)·Δt

    Args:
        positions : Shape (N, 3). Current positions r(0).
        velocities: Shape (N, 3). Initial velocities v(0).
        dt        : Time step Δt.

    Returns:
        np.ndarray: Shape (N, 3). Ghost previous positions.
    """
    return positions - velocities * dt


def verlet_step(
    positions: "np.ndarray",
    positions_prev: "np.ndarray",
    forces: "np.ndarray",
    dt: float,
    box
) -> "np.ndarray":
    """
    Advance positions by one timestep using the Verlet algorithm.

    r(t+Δt) = 2r(t) − r(t−Δt) + F(t)·Δt²   (since m=1, a=F)

    Applies Periodic Boundary Conditions after the update.

    Args:
        positions     : Shape (N, 3). r(t).
        positions_prev: Shape (N, 3). r(t−Δt).
        forces        : Shape (N, 3). F(t).
        dt            : Time step.
        box           : Box object for PBC.

    Returns:
        np.ndarray: Shape (N, 3). r(t+Δt).
    """
    positions_new = 2.0 * positions - positions_prev + forces * (dt ** 2)
    return positions_new % box.L    # Apply PBC


def estimate_velocities(
    positions_next: "np.ndarray",
    positions_prev: "np.ndarray",
    dt: float
) -> "np.ndarray":
    """
    Estimate velocities using the central difference formula.

    v(t) = [r(t+Δt) − r(t−Δt)] / (2Δt)

    This is second-order accurate in Δt.

    Args:
        positions_next: Shape (N, 3). r(t+Δt).
        positions_prev: Shape (N, 3). r(t−Δt).
        dt            : Time step.

    Returns:
        np.ndarray: Shape (N, 3). Estimated velocities v(t).
    """
    return (positions_next - positions_prev) / (2.0 * dt)