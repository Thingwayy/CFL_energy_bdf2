from mpi4py import MPI
import ufl
import basix.ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import time
from ufl import TrialFunctions, TestFunctions, inner, grad, cross

from error_analysis import compute_time_errors_and_rates
from initial_data_convergence import ue


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
    alpha,
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
        + inner(cross(m_h, delta_h_m_tilde), v_h) * dx_lumped \
        + alpha * inner(cross(m_h, cross(m_h, delta_h_m_tilde)), v_h) * dx_lumped \
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
        + inner(cross(m_h_hat, delta_h_m_tilde), v_h) * dx_lumped \
        + alpha * inner(cross(m_h_hat, cross(m_h_hat, delta_h_m_tilde)), v_h) * dx_lumped \
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


def main():
    start_time = time.perf_counter()

    # Experiment parameters. step_counts are coarse time steps; each is
    # compared against its 2x refined solve.
    step_counts = [60, 120, 240, 480, 960]
    order = 1
    mesh_n = 16
    total_time = 0.05
    alpha = 1.0

    error_records = []

    # Fixed spatial mesh and finite element spaces for the time-convergence test.
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.5, 0.5]), np.array([1.5, 1.5])],
        [mesh_n, mesh_n],
        mesh.CellType.quadrilateral,
    )
    dx_lumped = ufl.dx(metadata={"quadrature_rule": "GLL", "quadrature_degree": 2 * order - 1})

    u_el = basix.ufl.element("Q", domain.basix_cell(), order, shape=(3,))
    V = fem.functionspace(domain, u_el)
    el_w_mixed = basix.ufl.mixed_element([u_el, u_el])
    W = fem.functionspace(domain, el_w_mixed)
    solution_cache = {}

    def get_solutions(num_steps):
        # Cache histories because the fine solve for one pair becomes the
        # coarse solve for the next pair.
        if num_steps not in solution_cache:
            if domain.comm.rank == 0:
                print(f"\nSolving {num_steps} time steps")
            solution_cache[num_steps] = llg(
                num_steps,
                V,
                W,
                dx_lumped,
                total_time,
                alpha,
                domain.comm,
            )
        elif domain.comm.rank == 0:
            print(f"\nReusing {num_steps} time-step solution")

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

        previous_errors = error_records[-1] if error_records else None
        refinement_ratio = num_steps_coarse / step_counts[i - 1] if i > 0 else None
        error_record = compute_time_errors_and_rates(
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

    # Final summaries are printed after all pairwise comparisons finish.
    elapsed_time = time.perf_counter() - start_time

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
        print(f"  Elapsed time: {elapsed_time:.2f} s")


if __name__ == "__main__":
    main()
