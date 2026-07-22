import numpy as np
from newton_solver import (residual, jacobian, dR_dlambda, newton_solve,
                           solution_norm, signed_solution_norm_by_mode,)


def normalize_tangent(tu, tl):
    # normalizes a continuation tangent vector (tu, tl).
    nrm = np.sqrt(np.dot(tu, tu) + tl * tl)
    if nrm == 0.0:
        raise ValueError("Zero tangent encountered.")
    return tu / nrm, tl / nrm


def secant_tangent(u_prev, lam_prev, u_curr, lam_curr):
    # build normalized tangent from the secant between two continuation points
    tu = u_curr - u_prev
    tl = lam_curr - lam_prev
    return normalize_tangent(tu, tl)


def analytic_continuation_step(
    u_prev,
    lam_prev,
    dlam,
    n_interior,
    tol=1e-10,
    max_iter=30,
    verbose=False,
):
    """
    simple parameter continuation:
        lambda_new = lambda_prev + dlam
        initial guess = u_prev
    then apply fixed-lambda Newton
    """
    lam_new = lam_prev + dlam
    u_guess = u_prev.copy()
    u_new, info = newton_solve(
        u_guess,
        lam_new,
        n_interior=n_interior,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
    )
    return u_new, lam_new, info


def pseudo_arclength_corrector(
    u_pred,
    lam_pred,
    u_ref,
    lam_ref,
    tangent_u,
    tangent_lam,
    n_interior,
    tol=1e-10,
    max_iter=20,
    verbose=False,
):
    """
    Newton corrector for the augmented pseudo-arclength system:
        R(u, lam) = 0
        g(u, lam) = tangent_u^T (u-u_ref) + tangent_lam (lam-lam_ref) = 0
    - solves for both u and lambda
    """
    u = u_pred.copy()
    lam = float(lam_pred)

    # Newton iteration for the augmented system
    for k in range(max_iter):
        # compute the residual R(u, lam)
        R = residual(u, lam, n_interior)

        # jacobian of R with respect to u
        J = jacobian(u, lam, n_interior)

        # partial derivative of R with respect to lambda
        # lambda is treated as an unknown in the augmented Newton system
        Rlam = dR_dlambda(u)

        g = np.dot(tangent_u, (u - u_ref)) + tangent_lam * (lam - lam_ref)

        size = u.size
        A = np.zeros((size + 1, size + 1), dtype=float)
        b = np.zeros(size + 1, dtype=float)

        # populates linear system
        # top-left block with Jacobian dR/du
        A[:size, :size] = J

        # fill last column with dR/dlambda
        A[:size, size] = Rlam

        # last row: derivative of pseudo arc length constraint g
        A[size, :size] = tangent_u
        A[size, size] = tangent_lam

        # negative augmented residual
        b[:size] = -R
        b[size] = -g

        try:
            # gives the correction in the solution vector and in the continuation parameter
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # if augmented Jacobian is singular or if solve fails, report failure and exit
            return u, lam, {
                "converged": False,
                "iterations": k,
                "res_norm": np.sqrt(np.dot(R, R) + g * g),
                "step_norm": np.nan,
                "message": "Augmented Jacobian solve failed."
            }

        # update components
        du = delta[:size]
        dlam = delta[size]

        # apply Newton update
        u += du
        lam += dlam

        # update size in augmented (u, lambda)-space
        step_norm = np.sqrt(np.dot(du, du) + dlam * dlam)

        # augmented residual size
        res_norm = np.sqrt(np.dot(R, R) + g * g)

        if verbose:
            print(
                f"[PALC] iter={k:2d}  ||augR||={res_norm:.3e}  "
                f"||step||={step_norm:.3e}  lam={lam:.6f}"
            )

        # if the Newton update is small enough, accept the corrected point and we converge
        if step_norm < tol:
            return u, lam, {
                "converged": True,
                "iterations": k + 1,
                "res_norm": res_norm,
                "step_norm": step_norm,
                "message": "Pseudo-arclength corrector converged."
            }

    # if iteration limit is reached before convergence, report failure
    return u, lam, {
        "converged": False,
        "iterations": max_iter,
        "res_norm": res_norm,
        "step_norm": step_norm,
        "message": "Pseudo-arclength corrector hit max_iter."
    }


def initialize_branch_by_ac(
    u0,
    lam0,
    dlam,
    n_interior,
    tol=1e-10,
    max_iter=30,
    verbose=False,
):
    """
    initialize a continuation branch with:
      point 0: Newton solve at lam0
      point 1: one analytic continuation step to lam0 + dlam
    """
    u_start, info0 = newton_solve(
        u0,
        lam0,
        n_interior=n_interior,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
    )
    if not info0["converged"]:
        raise RuntimeError(
            f"Initial Newton solve failed at lambda={lam0}: {info0}"
        )

    u_next, lam_next, info1 = analytic_continuation_step(
        u_start,
        lam0,
        dlam,
        n_interior=n_interior,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
    )
    if not info1["converged"]:
        raise RuntimeError(
            f"Analytic continuation initialization failed at lambda={lam_next}: {info1}"
        )

    return (u_start, lam0, info0), (u_next, lam_next, info1)


def trace_branch_pseudo_arclength(
    u0,
    lam0,
    dlam_init,
    ds,
    n_steps,
    n_interior,
    mode_x,
    mode_y,
    lam_min=0,
    lam_max=60,
    tol=1e-10,
    max_newton_iter=30,
    max_corr_iter=20,
    verbose=False,
):
    """
    trace one branch using:
      - Newton solve at lam0
      - one analytic continuation step
      - repeated pseudo-arclength predictor/corrector steps
    """
    (u_prev, lam_prev, info_prev), (u_curr, lam_curr, info_curr) = initialize_branch_by_ac(
        u0=u0,
        lam0=lam0,
        dlam=dlam_init,
        n_interior=n_interior,
        tol=tol,
        max_iter=max_newton_iter,
        verbose=verbose,
    )

    lams = [lam_prev, lam_curr]
    norms = [solution_norm(u_prev), solution_norm(u_curr)]
    signed_norms = [
        signed_solution_norm_by_mode(u_prev, n_interior, mode_x, mode_y),
        signed_solution_norm_by_mode(u_curr, n_interior, mode_x, mode_y),
    ]
    solutions = [u_prev.copy(), u_curr.copy()]
    infos = [info_prev, info_curr]

    for step in range(n_steps):
        tangent_u, tangent_lam = secant_tangent(u_prev, lam_prev, u_curr, lam_curr)

        # predictor
        u_pred = u_curr + ds * tangent_u
        lam_pred = lam_curr + ds * tangent_lam

        # uses predictor point as the reference for the hyperplane constraint
        u_ref = u_pred.copy()
        lam_ref = lam_pred

        # corrector
        u_new, lam_new, info_new = pseudo_arclength_corrector(
            u_pred=u_pred,
            lam_pred=lam_pred,
            u_ref=u_ref,
            lam_ref=lam_ref,
            tangent_u=tangent_u,
            tangent_lam=tangent_lam,
            n_interior=n_interior,
            tol=tol,
            max_iter=max_corr_iter,
            verbose=verbose,
        )

        if not info_new["converged"]:
            if verbose:
                # PALC: pseudo arc length continuation
                print(f"[PALC] stopping at step {step}: corrector failed.")
            break

        if lam_new < lam_min - 1e-12 or lam_new > lam_max + 1e-12:
            if verbose:
                print(f"[PALC] stopping at step {step}: lambda={lam_new:.6f} out of range.")
            break

        lams.append(lam_new)
        norms.append(solution_norm(u_new))
        signed_norms.append(
            signed_solution_norm_by_mode(u_new, n_interior, mode_x, mode_y)
        )
        solutions.append(u_new.copy())
        infos.append(info_new)

        u_prev, lam_prev = u_curr, lam_curr
        u_curr, lam_curr = u_new, lam_new

    return {
        "lams": np.array(lams),
        "norms": np.array(norms),
        "signed_norms": np.array(signed_norms),
        "solutions": solutions,
        "infos": infos,
    }


def trace_multiple_branches(
    branch_specs,
    n_interior=30,
    default_dlam_init=0.2,
    default_ds=0.2,
    n_steps=100,
    lam_min=0,
    lam_max=60,
    tol=1e-10,
    max_newton_iter=30,
    max_corr_iter=20,
    verbose=False,
):
    """
    each branch spec should contain:
        {
            "name": str,
            "u0": ndarray,
            "lam0": float,
            "dlam_init": float,
            "ds": float
        }
    """
    branches = {}
    for spec in branch_specs:
        name = spec["name"]
        u0 = spec["u0"]
        lam0 = spec["lam0"]
        dlam_init = spec.get("dlam_init", default_dlam_init)
        ds = spec.get("ds", default_ds)
        mode_x = spec["mode_x"]
        mode_y = spec["mode_y"]

        if verbose:
            print(
                f"\n=== Tracing branch: {name} "
                f"from lam0={lam0:.6f}, dlam_init={dlam_init:.3f}, ds={ds:.3f} ==="
            )

        branch = trace_branch_pseudo_arclength(
            u0=u0,
            lam0=lam0,
            dlam_init=dlam_init,
            ds=ds,
            n_steps=n_steps,
            n_interior=n_interior,
            mode_x=mode_x,
            mode_y=mode_y,
            lam_min=lam_min,
            lam_max=lam_max,
            tol=tol,
            max_newton_iter=max_newton_iter,
            max_corr_iter=max_corr_iter,
            verbose=verbose,
        )
        branches[name] = branch

    return branches