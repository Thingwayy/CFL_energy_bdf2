from mpi4py import MPI
import ufl
import basix.ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import time
from ufl import TrialFunctions, TestFunctions, inner, grad, cross


def ue(x):
    # Smooth unit-length initial magnetization used for interpolation.
    comp1 = np.sin(np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + 1
    comp2 = np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) + 2
    comp3 = np.sin(np.pi * x[1])
    norm = np.sqrt(comp1**2 + comp2**2 + comp3**2)
    return np.vstack([comp1, comp2, comp3]) / norm


def _project_to_unit_sphere(m):
    # Nodal projection m <- m / |m| enforces the unit-length constraint.
    values = m.x.array
    norms = np.sqrt(values[0::3] ** 2 + values[1::3] ** 2 + values[2::3] ** 2)
    norms[norms < 1e-12] = 1.0
    values[0::3] /= norms
    values[1::3] /= norms
    values[2::3] /= norms
    m.x.scatter_forward()


def llg(
    num_steps,
    V,
    W,
    dx_lumped,
    total_time,
    comm=MPI.COMM_WORLD,
):
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

    solver_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
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
        petsc_options_prefix="llg_bdf1_",
    )

    # BDF2 step reused for all later time levels. m_h and m_h_prev are
    # Functions, so their updated vector values are used at each solve.
    bdf2_rhs = (1.0 / (2.0 * dt)) * inner(4.0 * m_h - m_h_prev, v_h) * dx_lumped
    bdf2_lhs = (3.0 / (2.0 * dt)) * inner(m_h_tilde, v_h) * dx_lumped \
        + inner(cross(m_h_hat, cross(m_h_hat, delta_h_m_tilde)), v_h) * dx_lumped \
        + inner(delta_h_m_tilde, bar_v_h) * dx_lumped \
        + inner(grad(m_h_tilde), grad(bar_v_h)) * dx_lumped
    bdf2_problem = LinearProblem(
        bdf2_lhs,
        bdf2_rhs,
        petsc_options=solver_options,
        petsc_options_prefix="llg_bdf2_",
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


def _validate_time_histories(solutions_coarse, solutions_fine, time_ratio, start_n):
    # The fine history must contain matching states at every coarse time.
    len_coarse = len(solutions_coarse)
    len_fine = len(solutions_fine)

    if len_coarse == 0:
        raise ValueError("solutions_coarse must not be empty")
    if time_ratio <= 0:
        raise ValueError("time_ratio must be positive")
    if start_n < 0 or start_n > len_coarse:
        raise ValueError("start_n is out of range")

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
    _validate_time_histories(solutions_coarse, solutions_fine, time_ratio, 0)

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
    step_counts = [60, 120, 240, 480, 960]
    order = 1
    mesh_n = 16
    total_time = 0.05

    error_records = []

    # Fixed spatial mesh and finite element spaces for the time-convergence test.
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.5, 0.5]), np.array([1.5, 1.5])],
        [mesh_n, mesh_n],
        mesh.CellType.quadrilateral,
    )
    dx_lumped = ufl.dx(metadata={"quadrature_rule": "GLL", "quadrature_degree": order})
    dx = ufl.Measure("dx", domain=domain)

    u_el = basix.ufl.element("Q", domain.basix_cell(), order, shape=(3,))
    V = fem.functionspace(domain, u_el)
    el_w_mixed = basix.ufl.mixed_element([u_el, u_el])
    W = fem.functionspace(domain, el_w_mixed)
    solution_cache = {}
    solve_times = {}

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
        error_u = error_record["final_L2"]
        error_u_H1 = error_record["final_H1"]
        error_L2H1 = error_record["L2H1"]
        error_LinfL2 = error_record["LinfL2"]
        max_linf_index = error_record["LinfL2_index"]
        compare_time = _elapsed_max(domain.comm, compare_start)
        error_records.append(error_record)

        if domain.comm.rank == 0:
            print(f"  Final-time L^2 error = {error_u:.3e}")
            print(f"  Final-time H^1 error = {error_u_H1:.3e}")
            print(
                f"  L^2(0,T;H^1) error (all)       = {error_L2H1:.3e} "
                f"(E/dt^2={error_L2H1 / dt_coarse**2:.3e})"
            )
            print(
                f"  L^infty(0,T;L^2) error (all)   = {error_LinfL2:.3e} "
                f"(E/dt^2={error_LinfL2 / dt_coarse**2:.3e})"
            )
            print(
                "    max at coarse step "
                f"{max_linf_index + 1}/{num_steps_coarse}, "
                f"t={(max_linf_index + 1) * dt_coarse:.4e}"
            )
            print(f"  comparison time: {compare_time:.2f} s")

    # Final summaries are printed after all pairwise comparisons finish.
    solve_total = sum(solve_times.values())
    total_time_elapsed = _elapsed_max(domain.comm, start_time)

    if MPI.COMM_WORLD.rank == 0:
        print("\n=== Final-time convergence rates ===")
        for i in range(1, len(step_counts)):
            rates = error_records[i]["rates"]
            rate_u = rates["final_L2"]
            rate_u_H1 = rates["final_H1"]
            print(f"  L^2 rate (steps {step_counts[i-1]}->{step_counts[i]}) = {rate_u:.4f}")
            print(f"  H^1 rate (steps {step_counts[i-1]}->{step_counts[i]}) = {rate_u_H1:.4f}")

        print("\n=== Full-history convergence rates ===")
        for i in range(1, len(step_counts)):
            rates = error_records[i]["rates"]
            rate_L2H1 = rates["L2H1"]
            rate_LinfL2 = rates["LinfL2"]
            print(f"  L^2(0,T;H^1) rate            = {rate_L2H1:.4f}")
            print(f"  L^infty(0,T;L^2) rate        = {rate_LinfL2:.4f}")

        print("\n=== Timing ===")
        print(f"  Total solve time: {solve_total:.2f} s")
        print(f"  Total elapsed time: {total_time_elapsed:.2f} s")


if __name__ == "__main__":
    main()
