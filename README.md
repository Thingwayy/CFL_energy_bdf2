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

The convergence results recorded below were obtained in the micromamba
environment `fenicsx-env` under Ubuntu 22.04.5 LTS on WSL. After opening the
Ubuntu terminal, activate the environment and enter the convergence directory:

```bash
micromamba activate fenicsx-env
cd /mnt/d/Desktop/LL_CFL_energy_BDF2/0_code/bdf2/convergence
```

## Numerical Scheme

The convergence scripts use:

- Q1 finite elements
- mass-lumped inner product through GLL quadrature
- one BDF1 startup step
- BDF2 time stepping afterward
- the single-cross LLG precession term
- the `alpha`-weighted double-cross damping term with `alpha = 1.0`
- nodal projection back to the unit sphere after each solve
- normalized extrapolation

Current conversion scope: only the four two- and three-dimensional scripts in
`convergence/` have been converted to the full LLG equation. In particular,
`stability_blowup/blowup_stability_bdf2.py` still uses the previous double-cross
heat-flow discretization and has not yet been converted to the full LLG form.

The BDF2 step uses the predictor

```text
m_hat^{n+1} = I_h((2 m^n - m^{n-1}) / |2 m^n - m^{n-1}|)
```

and then projects the computed `m_tilde^{n+1}` back to unit length.

## Running Convergence Tests

The commands used for the recorded runs were:

```bash
python 2d_initial_time_bdf2.py
python 2d_initial_space_bdf2.py
mpirun -n 2 python 3d_initial_time_bdf2.py
mpirun -n 4 python 3d_initial_space_bdf2.py
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

## Recorded Convergence Results

These are coarse-fine self-convergence errors, not errors with respect to an
exact solution. In the temporal tables, the row indexed by `N` compares the
solution computed with `N` time steps against the solution computed with `2N`
time steps. In the spatial tables, the row indexed by `n` compares the
`n^d` mesh against the `(2n)^d` mesh. Each rate is computed from two successive
error rows.

The abbreviated column names are:

- `final L2`: final-time `L^2` error
- `final H1`: final-time `H^1` error
- `L2(H1)`: full-history `L^2(0,T;H^1)` error
- `Linf(L2)`: full-history `L^\infty(0,T;L^2)` error

### Two-dimensional temporal self-convergence

Command:

```bash
python 2d_initial_time_bdf2.py
```

| `N` | final L2 | rate | final H1 | rate | L2(H1) | rate | Linf(L2) | rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60  | 4.066e-05 | -      | 2.804e-04 | -      | 5.362e-04 | -      | 6.040e-04 | -      |
| 120 | 9.983e-06 | 2.0262 | 6.823e-05 | 2.0389 | 1.488e-04 | 1.8492 | 1.838e-04 | 1.7165 |
| 240 | 2.473e-06 | 2.0133 | 1.684e-05 | 2.0190 | 3.934e-05 | 1.9195 | 5.218e-05 | 1.8165 |
| 480 | 6.153e-07 | 2.0068 | 4.181e-06 | 2.0094 | 1.012e-05 | 1.9592 | 1.413e-05 | 1.8843 |
| 960 | 1.535e-07 | 2.0034 | 1.042e-06 | 2.0047 | 2.565e-06 | 1.9797 | 3.713e-06 | 1.9284 |

Elapsed time: `166.42 s`.

### Three-dimensional temporal self-convergence

Command:

```bash
mpirun -n 2 python 3d_initial_time_bdf2.py
```

The fixed mesh is `12 x 12 x 12`, with 6,591 global vector degrees of freedom.

| `N` | final L2 | rate | final H1 | rate | L2(H1) | rate | Linf(L2) | rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60  | 5.251e-05 | -      | 3.532e-04 | -      | 6.574e-04 | -      | 7.108e-04 | -      |
| 120 | 1.290e-05 | 2.0253 | 8.619e-05 | 2.0349 | 1.828e-04 | 1.8464 | 2.168e-04 | 1.7132 |
| 240 | 3.195e-06 | 2.0134 | 2.129e-05 | 2.0175 | 4.832e-05 | 1.9199 | 6.154e-05 | 1.8167 |
| 480 | 7.949e-07 | 2.0070 | 5.289e-06 | 2.0088 | 1.242e-05 | 1.9599 | 1.666e-05 | 1.8852 |
| 960 | 1.982e-07 | 2.0036 | 1.318e-06 | 2.0044 | 3.148e-06 | 1.9801 | 4.371e-06 | 1.9302 |

Elapsed time: `164.30 s`.

At the shared `N = 120` and `N = 240` error levels, the two-process results
agree with the earlier one-process run to the printed precision.

### Two-dimensional spatial self-convergence

Command:

```bash
python 2d_initial_space_bdf2.py
```

All meshes use 10 time steps. The row indexed by `n` compares the `n x n` mesh
with the `2n x 2n` mesh.

| `n` | final L2 | rate | final H1 | rate | L2(H1) | rate | Linf(L2) | rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8  | 2.304e-03 | -      | 6.802e-02 | -      | 2.850e-02 | -      | 8.540e-03 | -      |
| 16 | 5.746e-04 | 2.0038 | 3.365e-02 | 1.0156 | 1.421e-02 | 1.0040 | 2.169e-03 | 1.9769 |
| 32 | 1.435e-04 | 2.0010 | 1.678e-02 | 1.0039 | 7.100e-03 | 1.0010 | 5.445e-04 | 1.9943 |
| 64 | 3.588e-05 | 2.0003 | 8.383e-03 | 1.0010 | 3.549e-03 | 1.0003 | 1.363e-04 | 1.9986 |

Elapsed time: `12.78 s`.

### Three-dimensional spatial self-convergence

Command:

```bash
mpirun -n 4 python 3d_initial_space_bdf2.py
```

All meshes use 10 time steps. The row indexed by `n` compares the
`n x n x n` mesh with the `2n x 2n x 2n` mesh.

| `n` | final L2 | rate | final H1 | rate | L2(H1) | rate | Linf(L2) | rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4  | 1.148e-02 | -      | 1.542e-01 | -      | 7.015e-02 | -      | 3.852e-02 | -      |
| 8  | 2.824e-03 | 2.0233 | 6.947e-02 | 1.1502 | 3.476e-02 | 1.0133 | 1.051e-02 | 1.8741 |
| 16 | 7.042e-04 | 2.0038 | 3.380e-02 | 1.0394 | 1.730e-02 | 1.0065 | 2.663e-03 | 1.9802 |

The largest mesh solved in this comparison is `32 x 32 x 32`, with 107,811
global vector degrees of freedom. Elapsed time: `16.57 s`.

For all spatial runs, the maximum full-history `L^2` difference occurred at the
first time step, `t = 5.0e-03`. For the temporal runs, this maximum occurred in
the first few time steps and moved toward `t = 0` under refinement.

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
