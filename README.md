# Lennard-Jones Molecular Dynamics Simulator

A 3D Lennard-Jones Molecular Dynamics (MD) simulator with periodic boundary conditions (PBC) and Verlet integration.

## Configuration Parameters

- **Time Step ($\Delta t$):** $10^{-15}$ ($10^{-15}\text{ s} = 1\text{ fs}$ in physical units)
- **Particles ($N$):** 100
- **Box Size ($L$):** $20.0\,\sigma$
- **Integration Algorithm:** Verlet Algorithm
- **Pair Potential:** Lennard-Jones 12-6 Potential with Minimum Image Convention

## Running the Simulation

```bash
python main.py
```

## Running Tests

```bash
python tests/test_lj.py
python tests/test_physics.py
```
