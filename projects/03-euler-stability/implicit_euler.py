import numpy as np
"""
    Euler step:
    x^{n+1} = x^n + dt * [(a - b*y^{n+1})*x^{n+1} - p*(x^{n+1})^2]
    y^{n+1} = y^n + dt * [(c*x^{n+1} - d)*y^{n+1} - q*(y^{n+1})^2]

Define u = (x, y) and the residual F(u) = u - u^n - dt * f(u) = 0,
which is solved by Newton iteration J_F(u^{(k)}) du = -F(u^{(k)}), u^{(k+1)} = u^{(k)} + du
"""

def residual(u_new, u_old, dt, params):
    # F(u_new) = u_new - u_old - dt * f(u_new)
    x, y = u_new
    a, b, c, d, p, q = params
    fx = (a - b * y) * x - p * x ** 2
    fy = (c * x - d) * y - q * y ** 2
    return np.array([x - u_old[0] - dt * fx, y - u_old[1] - dt * fy])


def jacobian(u_new, dt, params):
    """
    f1 = (a - b*y)*x - p*x^2
    f2 = (c*x - d)*y - q*y^2

    df1/dx = (a - b*y) - 2*p*x
    df1/dy = -b*x
    df2/dx = c*y
    df2/dy = (c*x - d) - 2*q*y

    J_F = I - dt * df/du
    """
    x, y = u_new
    a, b, c, d, p, q = params
    df1_dx = (a - b * y) - 2.0 * p * x
    df1_dy = -b * x
    df2_dx = c * y
    df2_dy = (c * x - d) - 2.0 * q * y

    J = np.array([
        [1.0 - dt * df1_dx, -dt * df1_dy],
        [     -dt * df2_dx, 1.0 - dt * df2_dy]])
    return J


def implicit_euler(x0, y0, dt, T, params, tol=1e-10, max_iter=50):
    n_steps = int(np.round(T / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    traj = np.zeros((n_steps + 1, 2))
    traj[0] = [x0, y0]
    iter_hist = np.zeros(n_steps, dtype=int)

    for n in range(n_steps):
        u_old = traj[n]

        # Newton initial guess: previous time step
        u_new = u_old.copy()

        for k in range(max_iter):
            F = residual(u_new, u_old, dt, params)
            res_norm = np.linalg.norm(F)
            if res_norm < tol:
                iter_hist[n] = k
                break

            J = jacobian(u_new, dt, params)
            try:
                du = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                # singular Jacobian: stop and return partial trajectory
                return t[: n + 1], traj[: n + 1], iter_hist[: n]

            u_new = u_new + du

            if np.linalg.norm(du) < tol:
                iter_hist[n] = k + 1
                break
        else:
            iter_hist[n] = max_iter

        traj[n + 1] = u_new

        if not np.all(np.isfinite(u_new)) or np.max(np.abs(u_new)) > 1e6:
            return t[: n + 2], traj[: n + 2], iter_hist[: n + 1]

    return t, traj, iter_hist