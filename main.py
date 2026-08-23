"""
main.py
=======
Lennard-Jones Molecular Dynamics Simulator — Entry Point

Runs a complete simulation and saves an energy plot.

Usage (from project root with md_sim environment active):
    python main.py
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from config import (
    N_PARTICLES, BOX_LENGTH, MIN_SEPARATION, MAX_VELOCITY,
    DT, N_STEPS, OUTPUT_INTERVAL, SAVE_PLOT, PLOT_FILENAME
)
from src.box         import Box
from src.particles   import init_positions, init_velocities, remove_com_velocity
from src.forces      import compute_forces_and_pe
from src.integrator  import bootstrap_prev_positions, verlet_step, estimate_velocities
from src.energy      import kinetic_energy, temperature, total_energy


def run_simulation():
    """Run the complete MD simulation. Returns energy arrays."""

    print("=" * 65)
    print("  Lennard-Jones MD Simulator")
    print("=" * 65)
    print(f"  N={N_PARTICLES}  L={BOX_LENGTH}σ  dt={DT}τ  steps={N_STEPS}")
    print("=" * 65)

    # ── Initialization ────────────────────────────────────────────────────────
    box            = Box(BOX_LENGTH)
    positions      = init_positions(N_PARTICLES, box, MIN_SEPARATION)
    velocities     = init_velocities(N_PARTICLES, MAX_VELOCITY)
    velocities     = remove_com_velocity(velocities)
    positions_prev = bootstrap_prev_positions(positions, velocities, DT)

    # ── Data storage ──────────────────────────────────────────────────────────
    steps              = np.arange(N_STEPS)
    kinetic_energies   = np.zeros(N_STEPS)
    potential_energies = np.zeros(N_STEPS)
    total_energies     = np.zeros(N_STEPS)

    # ── Main loop ─────────────────────────────────────────────────────────────
    print(f"\n  {'Step':>6}  {'KE':>10}  {'PE':>10}  {'Total E':>10}  {'T':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    t0 = time.time()
    for step in range(N_STEPS):

        forces, pe        = compute_forces_and_pe(positions, box)
        positions_new     = verlet_step(positions, positions_prev, forces, DT, box)
        velocities        = estimate_velocities(positions_new, positions_prev, DT)
        ke                = kinetic_energy(velocities)
        temp              = temperature(velocities)
        e_total           = total_energy(ke, pe)

        kinetic_energies[step]   = ke
        potential_energies[step] = pe
        total_energies[step]     = e_total

        if step % OUTPUT_INTERVAL == 0:
            print(f"  {step:6d}  {ke:10.4f}  {pe:10.4f}  {e_total:10.4f}  {temp:8.4f}")

        positions_prev = positions.copy()
        positions      = positions_new

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s  ({N_STEPS/elapsed:.0f} steps/s)")

    # ── Energy summary ────────────────────────────────────────────────────────
    e_mean  = np.mean(total_energies)
    e_drift = (total_energies[-1] - total_energies[0]) / abs(e_mean) * 100
    print(f"\n  Mean total energy : {e_mean:.4f} ε")
    print(f"  Energy drift      : {e_drift:.4f}%")
    if abs(e_drift) < 1.0:
        print("  ✓ Energy well conserved")
    else:
        print("  ⚠ Large drift — check DT or particle overlaps")

    return steps, kinetic_energies, potential_energies, total_energies


def plot_energy(steps, ke_arr, pe_arr, e_arr):
    """Save a four-panel energy plot: Energy vs Time AND Energy vs Step."""
    time_axis = steps * DT
    e_mean    = np.mean(e_arr)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"LJ-MD: N={N_PARTICLES}, L={BOX_LENGTH}σ, Δt={DT}",
        fontsize=14, fontweight='bold'
    )

    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    ax1.plot(time_axis, ke_arr,  'b-', lw=1.0, label='KE', alpha=0.8)
    ax1.plot(time_axis, pe_arr,  'r-', lw=1.0, label='PE', alpha=0.8)
    ax1.plot(time_axis, e_arr,   'k-', lw=2.0, label='Total E')
    ax1.axhline(e_mean, color='gray', ls='--', alpha=0.4)
    ax1.set_ylabel('Energy (ε)')
    ax1.set_xlabel('Time (τ)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title('Energy vs Time')

    ax2.plot(steps, ke_arr,  'b-', lw=1.0, label='KE', alpha=0.8)
    ax2.plot(steps, pe_arr,  'r-', lw=1.0, label='PE', alpha=0.8)
    ax2.plot(steps, e_arr,   'k-', lw=2.0, label='Total E')
    ax2.axhline(e_mean, color='gray', ls='--', alpha=0.4)
    ax2.set_ylabel('Energy (ε)')
    ax2.set_xlabel('Step')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_title('Energy vs Step')

    fluct = e_arr - e_mean
    ax3.plot(time_axis, fluct, 'k-', lw=1.0)
    ax3.axhline(0, color='gray', ls='--', alpha=0.4)
    ax3.set_xlabel('Time (τ)')
    ax3.set_ylabel('ΔE = E − ⟨E⟩ (ε)')
    ax3.set_title('Energy Fluctuation vs Time')
    ax3.grid(alpha=0.3)
    stats = f"⟨E⟩={e_mean:.3f}\nσ(E)={np.std(e_arr):.3f}"
    ax3.text(0.02, 0.95, stats, transform=ax3.transAxes,
             va='top', fontsize=9, bbox=dict(fc='wheat', alpha=0.5))

    ax4.plot(steps, fluct, 'k-', lw=1.0)
    ax4.axhline(0, color='gray', ls='--', alpha=0.4)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('ΔE = E − ⟨E⟩ (ε)')
    ax4.set_title('Energy Fluctuation vs Step')
    ax4.grid(alpha=0.3)
    ax4.text(0.02, 0.95, stats, transform=ax4.transAxes,
             va='top', fontsize=9, bbox=dict(fc='wheat', alpha=0.5))

    plt.tight_layout()
    if SAVE_PLOT:
        import os
        os.makedirs('output', exist_ok=True)
        plt.savefig(PLOT_FILENAME, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved → {PLOT_FILENAME}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    steps, ke_arr, pe_arr, e_arr = run_simulation()
    plot_energy(steps, ke_arr, pe_arr, e_arr)
    print("\n  Done! Open output/energy_vs_time.png to see your results.")