"""
Physics validation tests: energy conservation, force sum, COM velocity.
Run: python tests/test_physics.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.box import Box
from src.particles import init_positions, init_velocities, remove_com_velocity
from src.forces import compute_forces_and_pe
from src.integrator import bootstrap_prev_positions, verlet_step, estimate_velocities
from src.energy import kinetic_energy, total_energy
from config import BOX_LENGTH, N_PARTICLES, MIN_SEPARATION, MAX_VELOCITY, DT


def test_energy_conservation():
    box  = Box(BOX_LENGTH)
    pos  = init_positions(N_PARTICLES, box, MIN_SEPARATION)
    vel  = init_velocities(N_PARTICLES, MAX_VELOCITY)
    vel  = remove_com_velocity(vel)
    prev = bootstrap_prev_positions(pos, vel, DT)
    E    = []
    for _ in range(100):
        f, pe  = compute_forces_and_pe(pos, box)
        nxt    = verlet_step(pos, prev, f, DT, box)
        vel    = estimate_velocities(nxt, prev, DT)
        ke     = kinetic_energy(vel)
        E.append(total_energy(ke, pe))
        prev = pos.copy()
        pos  = nxt
    fluct = np.std(E[10:]) / abs(np.mean(E[10:]))
    assert fluct < 0.05, f"Energy fluctuation too large: {fluct*100:.2f}%"
    print(f"✓ Energy conserved: fluctuation = {fluct*100:.4f}%")


def test_total_force_zero():
    box  = Box(BOX_LENGTH)
    pos  = init_positions(10, box, MIN_SEPARATION)
    F, _ = compute_forces_and_pe(pos, box)
    total = np.sum(F, axis=0)
    assert np.allclose(total, 0, atol=1e-8), f"Total force not zero: {total}"
    print(f"✓ Total force = 0: {total}")


def test_com_removal():
    vel = init_velocities(100, 1.0)
    vel = remove_com_velocity(vel)
    com = np.mean(vel, axis=0)
    assert np.allclose(com, 0, atol=1e-12)
    print(f"✓ COM velocity after removal: {com}")


if __name__ == "__main__":
    print("=== Physics Validation Tests ===")
    test_total_force_zero()
    test_com_removal()
    test_energy_conservation()
    print("All physics tests passed ✓")