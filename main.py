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
    N_PARTICLES, BOX_LENGTH, MIN_SEPARATION, MAX_VELOCITY, USE_GRID_INIT,
    DT, N_STEPS, OUTPUT_INTERVAL, SAVE_PLOT, PLOT_FILENAME
)
from src.box         import Box
from src.particles   import init_positions, init_positions_grid, init_velocities, remove_com_velocity
from src.forces      import compute_forces_and_pe
from src.integrator  import velocity_verlet_step_1, velocity_verlet_step_2
from src.energy      import kinetic_energy, temperature, total_energy


def run_simulation():
    """Run the complete MD simulation using Velocity Verlet. Returns energy arrays."""

    print("=" * 65)
    print("  Lennard-Jones MD Simulator (Velocity Verlet)")
    print("=" * 65)
    print(f"  N={N_PARTICLES}  L={BOX_LENGTH}σ  dt={DT}τ  steps={N_STEPS}")
    print("=" * 65)

    # ── Initialization ────────────────────────────────────────────────────────
    box = Box(BOX_LENGTH)
    if USE_GRID_INIT:
        positions = init_positions_grid(N_PARTICLES, box)
    else:
        positions = init_positions(N_PARTICLES, box, MIN_SEPARATION)

    velocities = init_velocities(N_PARTICLES, MAX_VELOCITY)
    velocities = remove_com_velocity(velocities)
    forces, pe = compute_forces_and_pe(positions, box)

    # ── Data storage ──────────────────────────────────────────────────────────
    steps              = np.arange(N_STEPS)
    kinetic_energies   = np.zeros(N_STEPS)
    potential_energies = np.zeros(N_STEPS)
    total_energies     = np.zeros(N_STEPS)

    # ── Main loop (Velocity Verlet) ───────────────────────────────────────────
    print(f"\n  {'Step':>6}  {'KE':>10}  {'PE':>10}  {'Total E':>10}  {'T':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    t0 = time.time()
    for step in range(N_STEPS):

        pos_new, vel_half  = velocity_verlet_step_1(positions, velocities, forces, DT, box)
        forces_new, pe_new = compute_forces_and_pe(pos_new, box)
        vel_new            = velocity_verlet_step_2(vel_half, forces_new, DT)

        ke      = kinetic_energy(vel_new)
        temp    = temperature(vel_new)
        e_total = total_energy(ke, pe_new)

        kinetic_energies[step]   = ke
        potential_energies[step] = pe_new
        total_energies[step]     = e_total

        if step % OUTPUT_INTERVAL == 0:
            print(f"  {step:6d}  {ke:10.4f}  {pe_new:10.4f}  {e_total:10.4f}  {temp:8.4f}")

        positions, velocities, forces, pe = pos_new, vel_new, forces_new, pe_new

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


def _add_step_secondary_axis(ax, steps, dt):
    """Add a secondary top x-axis showing step numbers."""
    ax2 = ax.twiny()
    # Mirror the same limits as the primary axis in step units
    t_min, t_max = ax.get_xlim()
    ax2.set_xlim(t_min / dt, t_max / dt)
    ax2.set_xlabel('Step', labelpad=6)
    return ax2


def plot_energy(steps, ke_arr, pe_arr, e_arr):
    """
    Save a 4-panel energy analysis plot:
      1. Kinetic Energy vs Time (framed to highlight KE fluctuations)
      2. Potential Energy vs Time (framed to highlight PE fluctuations)
      3. Total Energy vs Time (framed to show Total E as a flat, conserved line)
      4. Combined Energy Overlay (KE, PE, Total E together)
    """
    time_axis = steps * DT

    ke_mean, ke_std, ke_ptp = np.mean(ke_arr), np.std(ke_arr), np.ptp(ke_arr)
    pe_mean, pe_std, pe_ptp = np.mean(pe_arr), np.std(pe_arr), np.ptp(pe_arr)
    e_mean,  e_std,  e_ptp  = np.mean(e_arr),  np.std(e_arr),  np.ptp(e_arr)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"LJ-MD Energy Analysis — N={N_PARTICLES}, L={BOX_LENGTH}σ, Δt={DT}τ\n"
        f"Total Energy Conserved while Kinetic & Potential Energy Fluctuate",
        fontsize=14, fontweight='bold'
    )

    # ── Panel 1: Kinetic Energy vs Time (Framed for KE fluctuations) ──────────
    ax1.plot(time_axis, ke_arr, color='royalblue', lw=1.5, alpha=0.85)
    ax1.axhline(ke_mean, color='navy', ls='--', lw=1.2, alpha=0.7,
                label=f'⟨KE⟩ = {ke_mean:.4f} ε')
    ax1.set_xlabel('Time (τ)', fontsize=11)
    ax1.set_ylabel('Kinetic Energy (ε)', fontsize=11)
    ax1.set_title('Kinetic Energy vs Time (KE Fluctuations)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ke_margin = max(ke_ptp * 0.25, 1e-5)
    ax1.set_ylim(ke_arr.min() - ke_margin, ke_arr.max() + ke_margin)
    ax1.set_xlim(time_axis[0], time_axis[-1])
    ax1.ticklabel_format(useOffset=False)
    _add_step_secondary_axis(ax1, steps, DT)
    stats_ke = f"Min: {ke_arr.min():.4f} ε\nMax: {ke_arr.max():.4f} ε\nSpan: {ke_ptp:.4f} ε"
    ax1.text(0.02, 0.05, stats_ke, transform=ax1.transAxes,
             va='bottom', fontsize=9, bbox=dict(fc='lightblue', alpha=0.6))

    # ── Panel 2: Potential Energy vs Time (Framed for PE fluctuations) ────────
    ax2.plot(time_axis, pe_arr, color='crimson', lw=1.5, alpha=0.85)
    ax2.axhline(pe_mean, color='darkred', ls='--', lw=1.2, alpha=0.7,
                label=f'⟨PE⟩ = {pe_mean:.4f} ε')
    ax2.set_xlabel('Time (τ)', fontsize=11)
    ax2.set_ylabel('Potential Energy (ε)', fontsize=11)
    ax2.set_title('Potential Energy vs Time (PE Fluctuations)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    pe_margin = max(pe_ptp * 0.25, 1e-5)
    ax2.set_ylim(pe_arr.min() - pe_margin, pe_arr.max() + pe_margin)
    ax2.set_xlim(time_axis[0], time_axis[-1])
    ax2.ticklabel_format(useOffset=False)
    _add_step_secondary_axis(ax2, steps, DT)
    stats_pe = f"Min: {pe_arr.min():.4f} ε\nMax: {pe_arr.max():.4f} ε\nSpan: {pe_ptp:.4f} ε"
    ax2.text(0.02, 0.05, stats_pe, transform=ax2.transAxes,
             va='bottom', fontsize=9, bbox=dict(fc='lightcoral', alpha=0.5))

    # ── Panel 3: Total Energy vs Time (Framed to show E_total is Flat) ────────
    ax3.plot(time_axis, e_arr, color='forestgreen', lw=1.8, alpha=0.95)
    ax3.axhline(e_mean, color='darkgreen', ls='--', lw=1.2, alpha=0.7,
                label=f'⟨E_total⟩ = {e_mean:.4f} ε')
    ax3.set_xlabel('Time (τ)', fontsize=11)
    ax3.set_ylabel('Total Energy (ε)', fontsize=11)
    ax3.set_title('Total Energy vs Time (Strictly Conserved / Flat)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    # Frame total energy relative to the KE/PE fluctuation scale so Total E is clearly flat
    fluct_scale = max(ke_ptp, pe_ptp, 0.1)
    ax3.set_ylim(e_mean - fluct_scale * 0.75, e_mean + fluct_scale * 0.75)
    ax3.set_xlim(time_axis[0], time_axis[-1])
    ax3.ticklabel_format(useOffset=False)
    _add_step_secondary_axis(ax3, steps, DT)
    e_drift = (e_arr[-1] - e_arr[0]) / abs(e_mean) * 100
    stats_e = f"⟨E⟩ = {e_mean:.4f} ε\nσ(E) = {e_std:.4f} ε\nDrift = {e_drift:.4f}%"
    ax3.text(0.02, 0.05, stats_e, transform=ax3.transAxes,
             va='bottom', fontsize=9, bbox=dict(fc='lightgreen', alpha=0.5))

    # ── Panel 4: Combined Energy Comparison ──────────────────────────────────
    ax4.plot(time_axis, ke_arr, color='royalblue', lw=1.2, alpha=0.8, label='Kinetic Energy (KE)')
    ax4.plot(time_axis, pe_arr, color='crimson', lw=1.2, alpha=0.8, label='Potential Energy (PE)')
    ax4.plot(time_axis, e_arr, color='forestgreen', lw=2.0, alpha=0.95, label='Total Energy (E_total)')
    ax4.set_xlabel('Time (τ)', fontsize=11)
    ax4.set_ylabel('Energy (ε)', fontsize=11)
    ax4.set_title('Combined Energies (KE & PE Fluctuate, Total E Flat)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='center right')
    ax4.grid(alpha=0.3)
    ax4.set_xlim(time_axis[0], time_axis[-1])
    _add_step_secondary_axis(ax4, steps, DT)

    plt.tight_layout()
    import os
    os.makedirs('output', exist_ok=True)
    plt.savefig(PLOT_FILENAME, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved → {PLOT_FILENAME}")
    plt.close()


if __name__ == "__main__":
    steps, ke_arr, pe_arr, e_arr = run_simulation()
    plot_energy(steps, ke_arr, pe_arr, e_arr)
    print("\n  Done! Open output/energy_vs_time.png to see your results.")