"""
Unit tests for the Lennard-Jones force and potential.
Run: python tests/test_lj.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.forces import lj_potential, lj_force_scalar, compute_forces_and_pe
from src.box import Box


def test_lj_zeros():
    assert abs(lj_potential(1.0)) < 1e-10, "U(σ) must be 0"
    print("✓ U(σ=1) = 0")

def test_lj_minimum():
    r_min = 2 ** (1/6)
    assert abs(lj_potential(r_min) - (-1.0)) < 1e-10, f"U(r_min) must be -1, got {lj_potential(r_min)}"
    print(f"✓ U(2^(1/6)σ) = -ε = -1")

def test_force_zero_at_minimum():
    r_min = 2 ** (1/6)
    assert abs(lj_force_scalar(r_min)) < 1e-10
    print("✓ F=0 at potential minimum")

def test_newton_third_law():
    box = Box(20.0)
    pos = np.array([[10.0, 10.0, 10.0], [11.5, 10.0, 10.0]])
    forces, _ = compute_forces_and_pe(pos, box)
    assert np.allclose(forces[0] + forces[1], 0, atol=1e-12)
    print("✓ Newton's 3rd Law: F_12 + F_21 = 0")

if __name__ == "__main__":
    print("=== LJ Unit Tests ===")
    test_lj_zeros()
    test_lj_minimum()
    test_force_zero_at_minimum()
    test_newton_third_law()
    print("All tests passed ✓")