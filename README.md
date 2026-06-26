# CFL Energy BDF2

This repository contains FEniCSx/DOLFINx scripts for testing a constraint-preserving BDF2 finite element scheme for sphere-valued heat-flow / LLG-type problems.

The code is organized for numerical experiments rather than as a reusable package. Most experiment scripts keep the BDF1/BDF2 scheme locally in the file so that the discretization can be read directly.

## Structure

```text
convergence/
  initial_data_convergence.py   # shared smooth 2D/3D initial data
  error_analysis.py             # time/space error and convergence-rate utilities
  2d_initial_time_bdf2.py       # 2D temporal self-convergence
  2d_initial_space_bdf2.py      # 2D spatial self-convergence
  3d_initial_time_bdf2.py       # 3D temporal self-convergence
  3d_initial_space_bdf2.py      # 3D spatial self-convergence

stability_blowup/
  initial_data_blowup.py        # blow-up initial data
  blowup_stability_bdf2.py      # energy stability and W^{1,infty} blow-up indicator
```

Generated text files such as `energy_*.txt` and `w1infty_*.txt` are ignored by Git.

## Environment

The scripts are intended to run in WSL/Linux with:

- Python
- NumPy
- mpi4py
- FEniCSx/DOLFINx
- PETSc/SLEPc stack with MUMPS support for the 2D direct solves

For 3D experiments, the scripts use iterative PETSc options to avoid expensive direct factorization.

## Numerical Scheme

The convergence scripts use:

- Q1 finite elements
- mass-lumped inner product through GLL quadrature
- one BDF1 startup step
- BDF2 time stepping afterward
- nodal projection back to the unit sphere after each solve
- normalized extrapolation

The BDF2 step uses the predictor

```text
m_hat^{n+1} = I_h((2 m^n - m^{n-1}) / |2 m^n - m^{n-1}|)
```

and then projects the computed `m_tilde^{n+1}` back to unit length.

## Running Convergence Tests

Run from the `convergence` directory:

```bash
cd convergence
python 2d_initial_time_bdf2.py
python 2d_initial_space_bdf2.py
python 3d_initial_time_bdf2.py
python 3d_initial_space_bdf2.py
```

For MPI runs:

```bash
mpirun -n 1 python 3d_initial_time_bdf2.py
```

The scripts print:

- final-time `L^2` and `H^1` errors
- full-history `L^2(0,T;H^1)` and `L^\infty(0,T;L^2)` errors
- convergence rates
- one total elapsed time

Main parameters are near the top of each `main()` function:

```python
step_counts = [...]
mesh_counts = [...]
mesh_n = ...
total_time = ...
num_steps = ...
```

## Running Blow-up / Stability Tests

Run from the `stability_blowup` directory:

```bash
cd stability_blowup
python blowup_stability_bdf2.py
```

The blow-up script uses the initial value

```text
m_0(x) = (x / |x| * sin(phi(|x|)), cos(phi(|x|)))

phi(r) = b r^2,  r <= 1
       = b,      r >= 1
```

with the default

```python
initial_b = 1.5 * np.pi
```

The script records:

- lumped discrete energy
- sampled `||grad m_h||_{L^\infty}`
- whether the energy is monotone
- the maximum `W^{1,\infty}` value and its time
- comparison with the mesh-dependent scale `2 sqrt(2) / h`

Two parameter blocks are left in the script:

```python
# Parameters for energy stability tests.
# mesh_counts = [64]
# total_time = 10.0
# num_steps_list = [10, 20, 40, 80]

# Parameters for blow-up detection.
mesh_counts = [8, 16, 32, 64]
total_time = 1.0
num_steps_list = [1000]
```

Switch between them by commenting/uncommenting the block you want to run.

## Notes

- Time convergence is computed by comparing `N` time steps with `2N` time steps.
- Space convergence is computed by comparing a mesh with its uniformly refined mesh.
- Blow-up indicators are numerical diagnostics. Reaching the discrete scale `2 sqrt(2) h^{-1}` is evidence of discrete blow-up behavior, not a proof of continuous finite-time blow-up.
- Large 3D tests can be slow; reduce `mesh_n` or the number of step counts for quick checks.
