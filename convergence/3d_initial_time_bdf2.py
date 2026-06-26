from mpi4py import MPI
import ufl
import basix.ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import time
from ufl import TrialFunctions, TestFunctions, inner, grad, cross


def ue(x):
    # Smooth unit-length initial magnetization on the 3D box.
    comp1 = np.sin(np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + np.cos(np.pi * x[2]) ** 2 + 1
    comp2 = np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + 2
    comp3 = np.sin(np.pi * x[1]) * np.cos(np.pi * x[2]) ** 2 + np.sin(np.pi * x[0])
    norm = np.sqrt(comp1**2 + comp2**2 + comp3**2)
    return np.vstack([comp1, comp2, comp3]) / norm


def _project_to_unit_sphere(m_h):
    # Nodal projection m_h <- m_h / |m_h| enforces the unit-length constraint.
    values = m_h.x.array
    norms = np.sqrt(values[0::3] ** 2 + values[1::3] ** 2 + values[2::3] ** 2)
    norms[norms < 1e-12] = 1.0
    values[0::3] /= norms
    values[1::3] /= norms
    values[2::3] /= norms
    m_h.x.scatter_forward()


def build_problem_spaces(mesh_n, order, comm):
    domain = mesh.create_box(
        comm,
        [np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5])],
        [mesh_n, mesh_n, mesh_n],
        mesh.CellType.hexahedron,
    )
    dx_lumped = ufl.dx(metadata={"quadrature_rule": "GLL", "quadrature_degree": 2 * order - 1})

    element = basix.ufl.element("Q", domain.basix_cell(), order, shape=(3,))
    V = fem.functionspace(domain, element)
    W = fem.functionspace(domain, basix.ufl.mixed_element([element, element]))

    return domain, V, W, dx_lumped


def llg(num_steps, V, W, dx_lumped, total_time, comm=MPI.COMM_WORLD):
    # Return the full time history [m^1, ..., m^N].
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    dt = total_time / num_steps

    m_h = fem.Function(V)
    m_h.interpolate(ue)
    m_h_prev = m_h.copy()
    m_h_hat = fem.Function(V)
    sol_history = []

    # Mixed solve variables: m_h_tilde is the unprojected magnetization,
    # delta_h_m_tilde represents its discrete Laplacian.
    (m_h_tilde, delta_h_m_tilde) = TrialFunctions(W)
    (v_h, bar_v_h) = TestFunctions(W)

    # 3D direct factorization is expensive, so use an iterative PETSc setup.
    solver_options = {
        "ksp_type": "gmres",
        "pc_type": "bjacobi",
        "sub_pc_type": "ilu",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
        "ksp_max_it": 2000,
        "ksp_error_if_not_converged": True,
    }

    # One BDF1 startup step from m^0 to m^1.
    bdf1_rhs = (1.0 / dt) * inner(m_h, v_h) * dx_lumped
    bdf1_lhs = (1.0 / dt) * inner(m_h_tilde, v_h) * dx_lumped \
        + inner(cross(m_h, cross(m_h, delta_h_m_tilde)), v_h) * dx_lumped \
        + inner(delta_h_m_tilde, bar_v_h) * dx_lumped \
        + inner(grad(m_h_tilde), grad(bar_v_h)) * dx_lumped
    bdf1_problem = LinearProblem(
        bdf1_lhs,
        bdf1_rhs,
        petsc_options=solver_options,
        petsc_options_prefix="llg3d_bdf1_",
    )

    # BDF2 step reused for all later time levels. m_h, m_h_prev, and
    # m_h_hat are Functions, so updated vector values enter every solve.
    bdf2_rhs = (1.0 / (2.0 * dt)) * inner(4.0 * m_h - m_h_prev, v_h) * dx_lumped
    bdf2_lhs = (3.0 / (2.0 * dt)) * inner(m_h_tilde, v_h) * dx_lumped \
        + inner(cross(m_h_hat, cross(m_h_hat, delta_h_m_tilde)), v_h) * dx_lumped \
        + inner(delta_h_m_tilde, bar_v_h) * dx_lumped \
        + inner(grad(m_h_tilde), grad(bar_v_h)) * dx_lumped
    bdf2_problem = LinearProblem(
        bdf2_lhs,
        bdf2_rhs,
        petsc_options=solver_options,
        petsc_options_prefix="llg3d_bdf2_",
    )

    for k in range(num_steps):
        if k == 0:
            mixed_solution = bdf1_problem.solve()
            m_h.interpolate(mixed_solution.sub(0))
            _project_to_unit_sphere(m_h)
        else:
            # Store m^n before overwriting m_h, then build normalized
            # extrapolation I_h((2m^n - m^{n-1}) / |2m^n - m^{n-1}|).
            m_h_n_values = m_h.x.array.copy()
            m_h_hat.x.array[:] = 2.0 * m_h.x.array - m_h_prev.x.array
            _project_to_unit_sphere(m_h_hat)

            mixed_solution = bdf2_problem.solve()

            m_h.interpolate(mixed_solution.sub(0))
            _project_to_unit_sphere(m_h)

            m_h_prev.x.array[:] = m_h_n_values
            m_h_prev.x.scatter_forward()

        sol_history.append(m_h.copy())

    return sol_history


def _validate_time_histories(solutions_coarse, solutions_fine, time_ratio):
    # The fine history must contain matching states at every coarse time.
    len_coarse = len(solutions_coarse)
    len_fine = len(solutions_fine)

    if len_coarse == 0:
        raise ValueError("solutions_coarse must not be empty")
    if time_ratio <= 0:
        raise ValueError("time_ratio must be positive")

    expected_len_fine = len_coarse * time_ratio
    if len_fine != expected_len_fine:
        raise ValueError(
            f"Time steps mismatch: coarse={len_coarse}, fine={len_fine}, "
            f"expected fine={expected_len_fine}"
        )


def _elapsed_max(comm, start_time):
    # Use the slowest MPI rank as the reported parallel runtime.
    return comm.allreduce(time.perf_counter() - start_time, op=MPI.MAX)


def compute_errors_and_rates(
    solutions_coarse,
    solutions_fine,
    domain,
    dt_coarse,
    time_ratio=2,
    previous_errors=None,
    refinement_ratio=None,
):
    # Compute final-time/full-history errors and optional convergence rates.
    _validate_time_histories(solutions_coarse, solutions_fine, time_ratio)

    dx = ufl.Measure("dx", domain=domain)
    diff = fem.Function(solutions_coarse[0].function_space)
    error_L2_sq_form = fem.form(inner(diff, diff) * dx)
    error_grad_sq_form = fem.form(inner(grad(diff), grad(diff)) * dx)

    L2H1_error_sq = 0.0
    LinfL2_error = 0.0
    max_index = 0
    final_L2_error = 0.0
    final_H1_error = 0.0
    final_index = len(solutions_coarse) - 1

    for n in range(len(solutions_coarse)):
        # coarse[n] is t=(n+1)dt_c; matching fine index is
        # (n+1)time_ratio - 1.
        sol_c = solutions_coarse[n]
        sol_f = solutions_fine[n * time_ratio + (time_ratio - 1)]
        diff.x.array[:] = sol_c.x.array - sol_f.x.array
        diff.x.scatter_forward()

        error_L2_sq = domain.comm.allreduce(fem.assemble_scalar(error_L2_sq_form), op=MPI.SUM)
        error_grad_sq = domain.comm.allreduce(fem.assemble_scalar(error_grad_sq_form), op=MPI.SUM)
        error_H1_sq = error_L2_sq + error_grad_sq
        error_L2 = np.sqrt(error_L2_sq)

        L2H1_error_sq += dt_coarse * error_H1_sq
        if error_L2 > LinfL2_error:
            LinfL2_error = error_L2
            max_index = n

        if n == final_index:
            final_L2_error = error_L2
            final_H1_error = np.sqrt(error_H1_sq)

    errors = {
        "final_L2": final_L2_error,
        "final_H1": final_H1_error,
        "L2H1": np.sqrt(L2H1_error_sq),
        "LinfL2": LinfL2_error,
        "LinfL2_index": max_index,
        "rates": None,
    }

    if previous_errors is not None:
        if refinement_ratio is None:
            raise ValueError("refinement_ratio is required when previous_errors is provided")
        errors["rates"] = {
            "final_L2": np.log(previous_errors["final_L2"] / errors["final_L2"]) / np.log(refinement_ratio),
            "final_H1": np.log(previous_errors["final_H1"] / errors["final_H1"]) / np.log(refinement_ratio),
            "L2H1": np.log(previous_errors["L2H1"] / errors["L2H1"]) / np.log(refinement_ratio),
            "LinfL2": np.log(previous_errors["LinfL2"] / errors["LinfL2"]) / np.log(refinement_ratio),
        }

    return errors


def main():
    start_time = time.perf_counter()

    # Experiment parameters. step_counts are coarse time steps; each is
    # compared against its 2x refined solve.
    step_counts = [120, 240]
    order = 1
    mesh_n = 12
    total_time = 0.05

    domain, V, W, dx_lumped = build_problem_spaces(mesh_n, order, MPI.COMM_WORLD)
    error_records = []
    solution_cache = {}
    solve_times = {}

    if domain.comm.rank == 0:
        global_dofs = V.dofmap.index_map.size_global * 3
        print(f"Mesh: {mesh_n}x{mesh_n}x{mesh_n}")
        print(f"Global vector DOFs: {global_dofs:,}")

    def get_solutions(num_steps):
        # Cache histories because the fine solve for one pair becomes the
        # coarse solve for the next pair.
        if num_steps not in solution_cache:
            if domain.comm.rank == 0:
                print(f"\nSolving {num_steps} time steps")
            solve_start = time.perf_counter()
            solution_cache[num_steps] = llg(
                num_steps,
                V,
                W,
                dx_lumped,
                total_time,
                domain.comm,
            )
            solve_times[num_steps] = _elapsed_max(domain.comm, solve_start)
            if domain.comm.rank == 0:
                print(f"  solve time: {solve_times[num_steps]:.2f} s")
        elif domain.comm.rank == 0:
            print(
                f"\nReusing {num_steps} time-step solution "
                f"(solve time was {solve_times[num_steps]:.2f} s)"
            )

        return solution_cache[num_steps]

    for i, num_steps_coarse in enumerate(step_counts):
        # Self-convergence: compare N time steps with 2N time steps.
        num_steps_fine = 2 * num_steps_coarse
        dt_coarse = total_time / num_steps_coarse
        solutions_coarse = get_solutions(num_steps_coarse)
        solutions_fine = get_solutions(num_steps_fine)

        if domain.comm.rank == 0:
            print(
                f"\nComparing {num_steps_coarse} steps with "
                f"{num_steps_fine} steps"
            )

        compare_start = time.perf_counter()
        previous_errors = error_records[-1] if error_records else None
        refinement_ratio = num_steps_coarse / step_counts[i - 1] if i > 0 else None
        error_record = compute_errors_and_rates(
            solutions_coarse,
            solutions_fine,
            domain,
            dt_coarse,
            time_ratio=2,
            previous_errors=previous_errors,
            refinement_ratio=refinement_ratio,
        )
        compare_time = _elapsed_max(domain.comm, compare_start)
        error_records.append(error_record)

        if domain.comm.rank == 0:
            max_linf_index = error_record["LinfL2_index"]
            print(f"  Final-time L^2 error = {error_record['final_L2']:.3e}")
            print(f"  Final-time H^1 error = {error_record['final_H1']:.3e}")
            print(
                f"  L^2(0,T;H^1) error (all)       = {error_record['L2H1']:.3e} "
                f"(E/dt^2={error_record['L2H1'] / dt_coarse**2:.3e})"
            )
            print(
                f"  L^infty(0,T;L^2) error (all)   = {error_record['LinfL2']:.3e} "
                f"(E/dt^2={error_record['LinfL2'] / dt_coarse**2:.3e})"
            )
            print(
                "    max at coarse step "
                f"{max_linf_index + 1}/{num_steps_coarse}, "
                f"t={(max_linf_index + 1) * dt_coarse:.4e}"
            )
            print(f"  comparison time: {compare_time:.2f} s")

    solve_total = sum(solve_times.values())
    total_time_elapsed = _elapsed_max(domain.comm, start_time)

    if MPI.COMM_WORLD.rank == 0:
        print("\n=== Final-time convergence rates ===")
        for i in range(1, len(step_counts)):
            rates = error_records[i]["rates"]
            print(f"  L^2 rate (steps {step_counts[i-1]}->{step_counts[i]}) = {rates['final_L2']:.4f}")
            print(f"  H^1 rate (steps {step_counts[i-1]}->{step_counts[i]}) = {rates['final_H1']:.4f}")

        print("\n=== Full-history convergence rates ===")
        for i in range(1, len(step_counts)):
            rates = error_records[i]["rates"]
            print(f"  L^2(0,T;H^1) rate            = {rates['L2H1']:.4f}")
            print(f"  L^infty(0,T;L^2) rate        = {rates['LinfL2']:.4f}")

        print("\n=== Timing ===")
        print(f"  Total solve time: {solve_total:.2f} s")
        print(f"  Total elapsed time: {total_time_elapsed:.2f} s")


if __name__ == "__main__":
    main()
