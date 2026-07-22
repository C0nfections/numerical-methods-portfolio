import numpy as np


def vec_to_grid(u_vec, n_interior):
    """
    convert interior unknown vector into a full grid with zero boundary values

    Parameters:
    u_vec: ndarray, shape (n_interior^2,)
        interior unknowns flattened to 1D
    n_interior: int
        num of interior points per spatial direction

    Returns:
    U: ndarray, shape (n_interior+2, n_interior+2)
        full grid including zero Dirichlet boundaries
    """
    N_full = n_interior + 2
    U = np.zeros((N_full, N_full), dtype=float)
    U[1:-1, 1:-1] = u_vec.reshape((n_interior, n_interior))
    return U


def grid_to_vec(U):
    # extract interior unknowns from full grid into a 1D vector
    return U[1:-1, 1:-1].reshape(-1)


def idx(i, j, n_interior):
    # map 2D interior index (i, j) to 1D flattened index
    return i * n_interior + j


def residual(u_vec, lam, n_interior):
    """
    residual for (nabla)^2 u + lam * u * (1 + u) = 0
    on the unit square with zero Dirichlet boundary conditions
    """
    h = 1.0 / (n_interior + 1)
    U = vec_to_grid(u_vec, n_interior)
    R = np.zeros((n_interior, n_interior), dtype=float)

    for i in range(1, n_interior + 1):
        for j in range(1, n_interior + 1):
            uij = U[i, j]
            lap = (U[i + 1, j] + U[i - 1, j] + U[i, j + 1] + U[i, j - 1] - 4.0 * uij) / h**2
            # 5-point stencil on the interior unknowns
            R[i - 1, j - 1] = lap + lam * uij * (1.0 + uij)

    return R.reshape(-1)


def jacobian(u_vec, lam, n_interior):
    # jacobian matrix dR/du for the discrete nonlinear system
    h = 1.0 / (n_interior + 1)
    U = vec_to_grid(u_vec, n_interior)

    size = n_interior * n_interior
    J = np.zeros((size, size), dtype=float)

    for i in range(n_interior):
        for j in range(n_interior):
            row = idx(i, j, n_interior)

            I = i + 1
            Jj = j + 1
            uij = U[I, Jj]

            # diagonal
            J[row, row] = -4.0 / h**2 + lam * (1.0 + 2.0 * uij)

            # neighbors
            if i > 0:
                J[row, idx(i - 1, j, n_interior)] = 1.0 / h**2  # left neighbor
            if i < n_interior - 1:
                J[row, idx(i + 1, j, n_interior)] = 1.0 / h**2  # right neighbor
            if j > 0:
                J[row, idx(i, j - 1, n_interior)] = 1.0 / h**2  # down neighbor
            if j < n_interior - 1:
                J[row, idx(i, j + 1, n_interior)] = 1.0 / h**2  # up neighbor

    return J


def dR_dlambda(u_vec):
    # partial derivative of residual with respect to lambda: dR/dlam = u * (1 + u)
    return u_vec * (1.0 + u_vec)


def solution_norm(u_vec):
    # euclidean norm of the interior solution vector
    return np.linalg.norm(u_vec, 2)


def signed_solution_norm(u_vec):
    """
    -signed norm for plotting positive and negative sub-branches
    -uses the sign of the mean value of the solution
    if the mean is >= 0, returns +||u||
    if the mean is < 0, returns -||u||
    """
    sgn = 1.0 if np.mean(u_vec) >= 0.0 else -1.0
    return sgn * np.linalg.norm(u_vec, 2)


def sine_mode_guess(n_interior, amplitude=0.1, mode_x=1, mode_y=1):
    # sine-mode initial guess satisfying zero Dirichlet BCs
    x = np.linspace(0.0, 1.0, n_interior + 2)
    y = np.linspace(0.0, 1.0, n_interior + 2)
    X, Y = np.meshgrid(x, y, indexing="ij")

    U = amplitude * np.sin(mode_x * np.pi * X) * np.sin(mode_y * np.pi * Y)
    return grid_to_vec(U)


def newton_solve(u0_vec, lam, n_interior=30, tol=1e-10, max_iter=30, verbose=False):
    """
    full Newton solve for fixed lambda; solves: R(u, lam) = 0

    Returns:
    u_vec: ndarray
    info: dict
    """
    u_vec = u0_vec.copy()

    for k in range(max_iter):
        R = residual(u_vec, lam, n_interior)
        J = jacobian(u_vec, lam, n_interior)

        res_norm = np.linalg.norm(R, 2)

        try:
            delta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            return u_vec, {
                "converged": False,
                "iterations": k,
                "res_norm": res_norm,
                "step_norm": np.nan,
                "message": "Jacobian solve failed."
            }

        step_norm = np.linalg.norm(delta, 2)
        u_vec = u_vec + delta

        if verbose:
            print(f"[Newton] iter={k:2d}  ||R||={res_norm:.3e}  ||du||={step_norm:.3e}")

        if step_norm < tol:
            return u_vec, {
                "converged": True,
                "iterations": k + 1,
                "res_norm": res_norm,
                "step_norm": step_norm,
                "message": "Newton converged."
            }

    return u_vec, {
        "converged": False,
        "iterations": max_iter,
        "res_norm": np.linalg.norm(residual(u_vec, lam, n_interior), 2),
        "step_norm": step_norm,
        "message": "Newton hit max_iter."
    }


def mode_vector(n_interior, mode_x, mode_y):
    # flattened sine mode vector on the interior grid
    x = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    y = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    X, Y = np.meshgrid(x, y, indexing="ij")
    M = np.sin(mode_x * np.pi * X) * np.sin(mode_y * np.pi * Y)
    return M.reshape(-1)


def signed_solution_norm_by_mode(u_vec, n_interior, mode_x, mode_y):
    # signed norm determined by projection onto a chosen eigenmode
    phi = mode_vector(n_interior, mode_x, mode_y)
    coeff = np.dot(u_vec, phi)
    sgn = 1.0 if coeff >= 0.0 else -1.0
    return sgn * np.linalg.norm(u_vec, 2)