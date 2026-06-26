from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem
from ufl import inner, grad


def _add_rates(errors, previous_errors, refinement_ratio):
    errors["rates"] = None
    if previous_errors is None:
        return errors

    if refinement_ratio is None:
        raise ValueError("refinement_ratio is required when previous_errors is provided")

    errors["rates"] = {
        "final_L2": np.log(previous_errors["final_L2"] / errors["final_L2"]) / np.log(refinement_ratio),
        "final_H1": np.log(previous_errors["final_H1"] / errors["final_H1"]) / np.log(refinement_ratio),
        "L2H1": np.log(previous_errors["L2H1"] / errors["L2H1"]) / np.log(refinement_ratio),
        "LinfL2": np.log(previous_errors["LinfL2"] / errors["LinfL2"]) / np.log(refinement_ratio),
    }
    return errors


def _accumulate_error_metrics(num_items, V_error, domain, dt, set_difference):
    dx = ufl.Measure("dx", domain=domain)
    diff = fem.Function(V_error)
    error_L2_sq_form = fem.form(inner(diff, diff) * dx)
    error_grad_sq_form = fem.form(inner(grad(diff), grad(diff)) * dx)

    L2H1_error_sq = 0.0
    LinfL2_error = 0.0
    max_index = 0
    final_L2_error = 0.0
    final_H1_error = 0.0
    final_index = num_items - 1

    for n in range(num_items):
        set_difference(n, diff)

        error_L2_sq = domain.comm.allreduce(fem.assemble_scalar(error_L2_sq_form), op=MPI.SUM)
        error_grad_sq = domain.comm.allreduce(fem.assemble_scalar(error_grad_sq_form), op=MPI.SUM)
        error_H1_sq = error_L2_sq + error_grad_sq
        error_L2 = np.sqrt(error_L2_sq)

        L2H1_error_sq += dt * error_H1_sq
        if error_L2 > LinfL2_error:
            LinfL2_error = error_L2
            max_index = n

        if n == final_index:
            final_L2_error = error_L2
            final_H1_error = np.sqrt(error_H1_sq)

    return {
        "final_L2": final_L2_error,
        "final_H1": final_H1_error,
        "L2H1": np.sqrt(L2H1_error_sq),
        "LinfL2": LinfL2_error,
        "LinfL2_index": max_index,
    }


def compute_time_errors_and_rates(
    solutions_coarse,
    solutions_fine,
    domain,
    dt_coarse,
    time_ratio=2,
    previous_errors=None,
    refinement_ratio=None,
):
    # Compare each coarse state with the matching fine state at the same time.
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

    def set_difference(n, diff):
        # coarse[n] is t=(n+1)dt_c; matching fine index is
        # (n+1)time_ratio - 1.
        sol_f = solutions_fine[n * time_ratio + (time_ratio - 1)]
        diff.x.array[:] = solutions_coarse[n].x.array - sol_f.x.array
        diff.x.scatter_forward()

    errors = _accumulate_error_metrics(
        len_coarse,
        solutions_coarse[0].function_space,
        domain,
        dt_coarse,
        set_difference,
    )
    return _add_rates(errors, previous_errors, refinement_ratio)


def compute_space_errors_and_rates(
    solutions_coarse,
    solutions_fine,
    V_fine,
    domain_fine,
    fine_cells,
    interpolation_data,
    dt,
    previous_errors=None,
    refinement_ratio=None,
):
    # Interpolate each coarse solution to the fine mesh before comparison.
    if len(solutions_coarse) == 0:
        raise ValueError("solutions_coarse must not be empty")
    if len(solutions_coarse) != len(solutions_fine):
        raise ValueError(
            f"Time history mismatch: coarse={len(solutions_coarse)}, "
            f"fine={len(solutions_fine)}"
        )

    m_coarse_on_fine = fem.Function(V_fine)

    def set_difference(n, diff):
        m_coarse_on_fine.interpolate_nonmatching(
            solutions_coarse[n],
            fine_cells,
            interpolation_data,
        )
        diff.x.array[:] = m_coarse_on_fine.x.array - solutions_fine[n].x.array
        diff.x.scatter_forward()

    errors = _accumulate_error_metrics(
        len(solutions_coarse),
        V_fine,
        domain_fine,
        dt,
        set_difference,
    )
    return _add_rates(errors, previous_errors, refinement_ratio)
