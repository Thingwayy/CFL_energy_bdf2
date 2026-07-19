from mpi4py import MPI
import ufl
import basix.ufl
from dolfinx import mesh, fem
from dolfinx.fem import create_interpolation_data
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import time
from ufl import TrialFunctions, TestFunctions, inner, grad, cross

from error_analysis import compute_space_errors_and_rates
from initial_data_convergence import ue


def _project_to_unit_sphere(m_h):
    # Nodal projection m_h <- m_h / |m_h| enforces the unit-length constraint.
    values = m_h.x.array
    norms = np.sqrt(values[0::3] ** 2 + values[1::3] ** 2 + values[2::3] ** 2)
    norms[norms < 1e-12] = 1.0
    values[0::3] /= norms
    values[1::3] /= norms
    values[2::3] /= norms
    m_h.x.scatter_forward()


def llg(num_steps, V, W, dx_lumped, total_time, alpha, comm=MPI.COMM_WORLD):
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

    # 3D direct factorization is expensive, so keep the original iterative
    # PETSc setup while using the same BDF1/BDF2 scheme as the 2D scripts.
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
        + inner(cross(m_h, delta_h_m_tilde), v_h) * dx_lumped \
        + alpha * inner(cross(m_h, cross(m_h, delta_h_m_tilde)), v_h) * dx_lumped \
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
        + inner(cross(m_h_hat, delta_h_m_tilde), v_h) * dx_lumped \
        + alpha * inner(cross(m_h_hat, cross(m_h_hat, delta_h_m_tilde)), v_h) * dx_lumped \
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


def main():
    start_time = time.perf_counter()

    # Experiment parameters. mesh_counts are coarse mesh subdivisions; each is
    # compared against its 2x refined mesh.
    mesh_counts = [4, 8, 16]
    order = 1
    total_time = 0.05
    num_steps = 10
    alpha = 1.0
    dt = total_time / num_steps

    solution_cache = {}
    error_records = []

    def get_solution_data(mesh_n):
        # Cache by mesh size because the fine mesh for one pair becomes the
        # coarse mesh for the next pair.
        if mesh_n not in solution_cache:
            if MPI.COMM_WORLD.rank == 0:
                print(f"\nSolving mesh {mesh_n}x{mesh_n}x{mesh_n} with {num_steps} time steps")

            domain = mesh.create_box(
                MPI.COMM_WORLD,
                [np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5])],
                [mesh_n, mesh_n, mesh_n],
                mesh.CellType.hexahedron,
            )
            dx_lumped = ufl.dx(metadata={"quadrature_rule": "GLL", "quadrature_degree": 2 * order - 1})
            element = basix.ufl.element("Q", domain.basix_cell(), order, shape=(3,))
            V = fem.functionspace(domain, element)
            W = fem.functionspace(domain, basix.ufl.mixed_element([element, element]))

            history = llg(num_steps, V, W, dx_lumped, total_time, alpha, domain.comm)

            solution_cache[mesh_n] = {
                "domain": domain,
                "V": V,
                "history": history,
            }
            if MPI.COMM_WORLD.rank == 0:
                global_dofs = V.dofmap.index_map.size_global * 3
                print(f"  global vector DOFs: {global_dofs:,}")
        elif MPI.COMM_WORLD.rank == 0:
            print(f"\nReusing mesh {mesh_n}x{mesh_n}x{mesh_n} solution")

        return solution_cache[mesh_n]

    for i, mesh_n_coarse in enumerate(mesh_counts):
        # Spatial self-convergence: compare mesh h with mesh h/2.
        mesh_n_fine = 2 * mesh_n_coarse
        coarse_data = get_solution_data(mesh_n_coarse)
        fine_data = get_solution_data(mesh_n_fine)

        domain_fine = fine_data["domain"]
        V_coarse = coarse_data["V"]
        V_fine = fine_data["V"]
        fine_cells = np.arange(
            domain_fine.topology.index_map(domain_fine.topology.dim).size_local,
            dtype=np.int32,
        )
        interpolation_data = create_interpolation_data(
            V_fine,
            V_coarse,
            fine_cells,
            padding=1e-6,
        )

        if domain_fine.comm.rank == 0:
            print(f"\nComparing mesh {mesh_n_coarse} with mesh {mesh_n_fine}")

        previous_errors = error_records[-1] if error_records else None
        refinement_ratio = mesh_n_coarse / mesh_counts[i - 1] if i > 0 else None
        error_record = compute_space_errors_and_rates(
            coarse_data["history"],
            fine_data["history"],
            V_fine,
            domain_fine,
            fine_cells,
            interpolation_data,
            dt,
            previous_errors=previous_errors,
            refinement_ratio=refinement_ratio,
        )
        error_records.append(error_record)

        if domain_fine.comm.rank == 0:
            max_linf_index = error_record["LinfL2_index"]
            print(f"  Final-time L^2 error = {error_record['final_L2']:.3e}")
            print(f"  Final-time H^1 error = {error_record['final_H1']:.3e}")
            print(f"  L^2(0,T;H^1) error   = {error_record['L2H1']:.3e}")
            print(f"  L^infty(0,T;L^2) error = {error_record['LinfL2']:.3e}")
            print(
                "    max at time step "
                f"{max_linf_index + 1}/{num_steps}, "
                f"t={(max_linf_index + 1) * dt:.4e}"
            )

    elapsed_time = time.perf_counter() - start_time

    if MPI.COMM_WORLD.rank == 0:
        print("\n=== Final-time spatial convergence rates ===")
        for i in range(1, len(mesh_counts)):
            rates = error_records[i]["rates"]
            print(
                f"  L^2 rate (mesh {mesh_counts[i-1]}->{mesh_counts[i]}) "
                f"= {rates['final_L2']:.4f}"
            )
            print(
                f"  H^1 rate (mesh {mesh_counts[i-1]}->{mesh_counts[i]}) "
                f"= {rates['final_H1']:.4f}"
            )

        print("\n=== Full-history spatial convergence rates ===")
        for i in range(1, len(mesh_counts)):
            rates = error_records[i]["rates"]
            print(f"  L^2(0,T;H^1) rate     = {rates['L2H1']:.4f}")
            print(f"  L^infty(0,T;L^2) rate = {rates['LinfL2']:.4f}")

        print("\n=== Timing ===")
        print(f"  Elapsed time: {elapsed_time:.2f} s")


if __name__ == "__main__":
    main()
