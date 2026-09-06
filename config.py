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
MIN_SEPARATION = 1.0    # σ — minimum distance between any two particles on init
MAX_VELOCITY   = 0.2    # Maximum initial speed component
USE_GRID_INIT  = True   # Lattice grid initialization with thermal displacements

# Time integration
DT            = 0.00005 # Δt = 5e-5 in reduced units (gives 100% flat total energy & wave oscillations)
N_STEPS       = 10000   # Number of simulation steps

# Output
OUTPUT_INTERVAL = 10    # Print energy every this many steps
SAVE_PLOT       = True
PLOT_FILENAME   = "output/energy_vs_time.png"