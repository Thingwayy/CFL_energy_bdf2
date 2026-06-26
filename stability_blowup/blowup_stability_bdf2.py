from mpi4py import MPI
import ufl
import basix.ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import time
from ufl import TrialFunctions, TestFunctions, inner, grad, cross

from initial_data_blowup import m_ini


def _project_to_unit_sphere(m_h):
    # Nodal projection m_h <- m_h / |m_h|.
    values = m_h.x.array
    norms = np.sqrt(values[0::3] ** 2 + values[1::3] ** 2 + values[2::3] ** 2)

    norms[norms < 1e-12] = 1.0
    values[0::3] /= norms
    values[1::3] /= norms
    values[2::3] /= norms
    m_h.x.scatter_forward()


def assemble_energy(energy_form, comm):
    local_energy = fem.assemble_scalar(energy_form)
    return comm.allreduce(local_energy, op=MPI.SUM)


def compute_w1inf_norm(grad_m_infty, grad_expression, comm):
    # DG sampled maximum of |grad m_h|; this is not an exact continuous maximum.
    grad_m_infty.interpolate(grad_expression)
    local_values = grad_m_infty.x.array
    local_max = np.max(local_values) if local_values.size else 0.0
    return comm.allreduce(local_max, op=MPI.MAX)


def solve_blowup_history(
    num_steps,
    domain,
    V,
    W,
    V_infty,
    dx_lumped,
    total_time,
    initial_b,
    print_every,
    comm=MPI.COMM_WORLD,
):
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    dt = total_time / num_steps

    m_h = fem.Function(V)
    m_h.interpolate(lambda x: m_ini(x, b=initial_b))
    _project_to_unit_sphere(m_h)

    m_h_prev = m_h.copy()
    m_h_hat = fem.Function(V)

    energy_form = fem.form(0.5 * inner(grad(m_h), grad(m_h)) * dx_lumped)
    grad_m_infty = fem.Function(V_infty)
    grad_expr = ufl.sqrt(inner(grad(m_h), grad(m_h)))

    # DOLFINx versions differ between a method and a property here.
    interpolation_points = V_infty.element.interpolation_points
    interpolation_points = interpolation_points() if callable(interpolation_points) else interpolation_points
    grad_expression = fem.Expression(grad_expr, interpolation_points)

    time_history = np.zeros(num_steps + 1)
    energy_history = np.zeros(num_steps + 1)
    w1inf_history = np.zeros(num_steps + 1)

    time_history[0] = 0.0
    energy_history[0] = assemble_energy(energy_form, comm)
    w1inf_history[0] = compute_w1inf_norm(grad_m_infty, grad_expression, comm)

    # Mixed solve variables: m_h_tilde is the unprojected magnetization,
    # delta_h_m_tilde represents its discrete Laplacian.
    (m_h_tilde, delta_h_m_tilde) = TrialFunctions(W)
    (v_h, bar_v_h) = TestFunctions(W)

    # This experiment is 2D, so a direct solve is affordable and avoids
    # iterative residuals masking small energy changes.
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
        petsc_options_prefix="blowup_bdf1_",
    )

    # Paper BDF2 step with normalized extrapolation:
    # m_h_hat = I_h((2m_h^n - m_h^{n-1}) / |2m_h^n - m_h^{n-1}|).
    bdf2_rhs = (1.0 / (2.0 * dt)) * inner(4.0 * m_h - m_h_prev, v_h) * dx_lumped
    bdf2_lhs = (3.0 / (2.0 * dt)) * inner(m_h_tilde, v_h) * dx_lumped \
        + inner(cross(m_h_hat, cross(m_h_hat, delta_h_m_tilde)), v_h) * dx_lumped \
        + inner(delta_h_m_tilde, bar_v_h) * dx_lumped \
        + inner(grad(m_h_tilde), grad(bar_v_h)) * dx_lumped
    bdf2_problem = LinearProblem(
        bdf2_lhs,
        bdf2_rhs,
        petsc_options=solver_options,
        petsc_options_prefix="blowup_bdf2_",
    )

    for k in range(num_steps):
        t = (k + 1) * dt

        if k == 0:
            mixed_solution = bdf1_problem.solve()
            m_h.interpolate(mixed_solution.sub(0))
            _project_to_unit_sphere(m_h)
        else:
            # Store m^n before overwriting m_h, then update m_h_hat.
            m_h_n_values = m_h.x.array.copy()
            m_h_hat.x.array[:] = 2.0 * m_h.x.array - m_h_prev.x.array
            _project_to_unit_sphere(m_h_hat)

            mixed_solution = bdf2_problem.solve()
            m_h.interpolate(mixed_solution.sub(0))
            _project_to_unit_sphere(m_h)

            m_h_prev.x.array[:] = m_h_n_values
            m_h_prev.x.scatter_forward()

        time_history[k + 1] = t
        energy_history[k + 1] = assemble_energy(energy_form, comm)
        w1inf_history[k + 1] = compute_w1inf_norm(grad_m_infty, grad_expression, comm)

        if comm.rank == 0 and ((k + 1) % print_every == 0 or k + 1 == num_steps):
            print(
                f"Step {k + 1}/{num_steps}, t={t:.6e}, "
                f"E={energy_history[k + 1]:.6e}, "
                f"||grad m||_inf={w1inf_history[k + 1]:.6e}"
            )

    return time_history, energy_history, w1inf_history


def save_history(filename, *columns):
    data = np.column_stack(columns)
    np.savetxt(filename, data, fmt="%.10e")


def main():
    start_time = time.perf_counter()

    order = 1
    initial_b = 1.5 * np.pi
    print_every = 50

    # Parameters for energy stability tests.
    # mesh_counts = [64]
    # total_time = 10.0
    # num_steps_list = [10, 20, 40, 80]

    # Parameters for blow-up detection.
    mesh_counts = [8, 16, 32, 64]
    total_time = 1.0
    num_steps_list = [1000]

    for mesh_n in mesh_counts:
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([-1.0, -1.0]), np.array([1.0, 1.0])],
            [mesh_n, mesh_n],
            mesh.CellType.quadrilateral,
        )
        dx_lumped = ufl.dx(metadata={"quadrature_rule": "GLL", "quadrature_degree": 2 * order - 1})

        element = basix.ufl.element("Q", domain.basix_cell(), order, shape=(3,))
        V = fem.functionspace(domain, element)
        W = fem.functionspace(domain, basix.ufl.mixed_element([element, element]))

        # DG1 samples |grad m_h| at each cell's nodal interpolation points.
        V_infty = fem.functionspace(domain, ("DG", 1))

        if domain.comm.rank == 0:
            global_dofs = V.dofmap.index_map.size_global * 3
            h = 2.0 / mesh_n
            print(f"\nMesh: {mesh_n}x{mesh_n}, h={h:.6e}")
            print(f"Global vector DOFs: {global_dofs:,}")
            print("-" * 40)

        for num_steps in num_steps_list:
            dt = total_time / num_steps
            if domain.comm.rank == 0:
                print(f"\nSolving {num_steps} time steps, dt={dt:.6e}")

            solve_start = time.perf_counter()
            time_hist, energy_hist, w1inf_hist = solve_blowup_history(
                num_steps,
                domain,
                V,
                W,
                V_infty,
                dx_lumped,
                total_time,
                initial_b,
                print_every,
                domain.comm,
            )
            solve_time = domain.comm.allreduce(time.perf_counter() - solve_start, op=MPI.MAX)

            if domain.comm.rank == 0:
                h = 2.0 / mesh_n
                wmax_index = int(np.argmax(w1inf_hist))
                wmax = w1inf_hist[wmax_index]
                t_wmax = time_hist[wmax_index]
                blowup_scale = 2.0 * np.sqrt(2.0) / h

                energy_diff = np.diff(energy_hist)
                positive_diff = energy_diff[energy_diff > 1e-10]
                max_positive_diff = np.max(positive_diff) if positive_diff.size else 0.0
                monotone = positive_diff.size == 0

                energy_drop_rate = -np.diff(energy_hist) / np.diff(time_hist)
                drop_index = int(np.argmax(energy_drop_rate))
                t_fast_drop = time_hist[drop_index + 1]

                tag = f"mesh{mesh_n}_steps{num_steps}"
                save_history(f"energy_{tag}.txt", time_hist, energy_hist)
                save_history(f"w1infty_{tag}.txt", time_hist, w1inf_hist)

                status = "yes" if monotone else "no"
                print(
                    f"  E(0)={energy_hist[0]:.6e}, E(T)={energy_hist[-1]:.6e}, "
                    f"monotone={status}, max dE+={max_positive_diff:.3e}"
                )
                print(
                    f"  max ||grad m||_inf={wmax:.6e} at t={t_wmax:.6e}, "
                    f"2sqrt(2)/h={blowup_scale:.6e}, ratio={wmax / blowup_scale:.4f}"
                )
                print(f"  fastest energy drop near t={t_fast_drop:.6e}")
                print(f"  solve time: {solve_time:.2f} s")

    total_elapsed = MPI.COMM_WORLD.allreduce(time.perf_counter() - start_time, op=MPI.MAX)
    if MPI.COMM_WORLD.rank == 0:
        print(f"\nTotal computation time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
