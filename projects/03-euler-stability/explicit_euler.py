import numpy as np

def rhs(state, params):
    x, y = state
    a, b, c, d, p, q = params
    dxdt = (a - b * y) * x - p * x ** 2
    dydt = (c * x - d) * y - q * y ** 2
    return np.array([dxdt, dydt])


def explicit_euler(x0, y0, dt, T, params):
    # u^{n+1} = u^n + dt * f(u^n)
    n_steps = int(np.round(T / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    traj = np.zeros((n_steps + 1, 2))
    traj[0] = [x0, y0]

    for n in range(n_steps):
        traj[n + 1] = traj[n] + dt * rhs(traj[n], params)

        # stop if trajectory has clearly blown up
        if not np.all(np.isfinite(traj[n + 1])) or np.max(np.abs(traj[n + 1])) > 1e6:
            t = t[: n + 2]
            traj = traj[: n + 2]
            break

    return t, traj