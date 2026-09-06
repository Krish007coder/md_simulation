"""
particles.py — Particle initialization.

Functions:
    init_positions     — Place N particles with minimum separation
    init_velocities    — Assign random initial velocities
    remove_com_velocity — Remove center-of-mass drift
    particle_summary   — Print diagnostic summary
"""

import numpy as np


def init_positions_grid(N: int, box, disp_amplitude: float = 0.15) -> "np.ndarray":
    """
    Place N particles on a 3D simple cubic lattice with small random thermal displacements.

    Prevents particle overlaps at initialization while creating natural thermal vibrations
    that produce clear, wave-like oscillating ups-and-downs in Kinetic & Potential Energy.
    """
    n_side = int(np.ceil(N ** (1/3)))
    spacing = box.L / n_side
    positions = []
    np.random.seed(42)
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if len(positions) < N:
                    disp = np.random.uniform(-disp_amplitude, disp_amplitude, size=3)
                    pos_i = [
                        ((ix + 0.5) * spacing + disp[0]) % box.L,
                        ((iy + 0.5) * spacing + disp[1]) % box.L,
                        ((iz + 0.5) * spacing + disp[2]) % box.L
                    ]
                    positions.append(pos_i)
    print(f"  Placed {N} particles on {n_side}x{n_side}x{n_side} lattice with thermal vibrations...")
    return np.array(positions, dtype=np.float64)


def init_positions(N: int, box, min_sep: float) -> "np.ndarray":
    """
    Place N particles in the box using rejection sampling.

    Generates random candidate positions and accepts them only if
    they are at least min_sep away from all previously placed particles.

    Args:
        N       : Number of particles.
        box     : Box object (uses box.L).
        min_sep : Minimum allowed distance between any two particles.

    Returns:
        positions: Shape (N, 3). All positions inside [0, box.L).

    Raises:
        RuntimeError: If a particle cannot be placed after 10,000 attempts.
    """
    positions    = np.zeros((N, 3), dtype=np.float64)
    max_attempts = 10_000

    for i in range(N):
        placed   = False
        attempts = 0

        while not placed:
            candidate = np.random.uniform(0.0, box.L, size=3)

            if i == 0:
                positions[0] = candidate
                placed = True
            else:
                too_close = False
                for j in range(i):
                    disp = candidate - positions[j]
                    disp = disp - box.L * np.round(disp / box.L)
                    if np.linalg.norm(disp) < min_sep:
                        too_close = True
                        break
                if not too_close:
                    positions[i] = candidate
                    placed = True

            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Cannot place particle {i} after {max_attempts} attempts. "
                    "Increase box size or reduce min_sep."
                )

        if (i + 1) % 20 == 0:
            print(f"  Placed {i+1}/{N} particles...")

    return positions


def init_velocities(N: int, max_vel: float) -> "np.ndarray":
    """
    Assign random initial velocities from Uniform[-max_vel, +max_vel].

    Args:
        N       : Number of particles.
        max_vel : Maximum speed in any one direction.

    Returns:
        velocities: Shape (N, 3).
    """
    return np.random.uniform(-max_vel, max_vel, size=(N, 3))


def remove_com_velocity(velocities: "np.ndarray") -> "np.ndarray":
    """
    Subtract the center-of-mass velocity from all particles.

    After this operation, the mean velocity is zero (no bulk drift).

    Args:
        velocities: Shape (N, 3). Modified in-place.

    Returns:
        velocities: Same array with COM velocity removed.
    """
    com = np.mean(velocities, axis=0)   # shape (3,)
    velocities -= com                   # broadcast: subtract from every particle
    return velocities


def particle_summary(positions: "np.ndarray",
                     velocities: "np.ndarray",
                     box) -> None:
    """Print a diagnostic summary of the particle state."""
    N = len(positions)
    print(f"\n{'='*50}")
    print("Particle System Summary")
    print(f"{'='*50}")
    print(f"N = {N}, L = {box.L} σ")
    print(f"COM velocity: {np.mean(velocities, axis=0)}")
    speeds = np.linalg.norm(velocities, axis=1)
    print(f"Speed range: {speeds.min():.3f} — {speeds.max():.3f}")
    print(f"{'='*50}\n")