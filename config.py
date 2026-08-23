"""
config.py
=========
All simulation parameters in one place.
Edit this file to change box size, timestep, particle count, etc.

All units are reduced Lennard-Jones units: σ = ε = m = 1
"""

# System
N_PARTICLES   = 100
BOX_LENGTH    = 20.0    # σ

# LJ parameters
SIGMA         = 1.0
EPSILON       = 1.0
MASS          = 1.0

# Initialization
MIN_SEPARATION = 0.9    # σ — minimum distance between any two particles on init
MAX_VELOCITY   = 1.0    # Maximum initial speed component

# Time integration
DT            = 1e-6    # Δt = 10⁻⁶ in reduced time units
N_STEPS       = 1000    # Number of simulation steps

# Output
OUTPUT_INTERVAL = 10    # Print energy every this many steps
SAVE_PLOT       = True
PLOT_FILENAME   = "output/energy_vs_time.png"