import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from explicit_euler import explicit_euler
from implicit_euler import implicit_euler


def add_arrows(ax, x, y, n_arrows=4, color=None):
    if len(x) < 4:
        return
    idx = np.linspace(1, len(x) - 2, n_arrows).astype(int)
    for k in idx:
        dx = x[k + 1] - x[k]
        dy = y[k + 1] - y[k]
        if dx == 0 and dy == 0:
            continue
        ax.annotate(
            "",
            xy=(x[k] + dx, y[k] + dy),
            xytext=(x[k], y[k]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )


def style_phase_axes(ax):
    ax.set_xlabel(r"$x$ (prey)")
    ax.set_ylabel(r"$y$ (predator)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")


OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# part c) parameters
params_c = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0)   # a=b=c=d=1, p=q=0

# initial conditions in (0, 10) x (0, 10)
ICs = [
    (1.5, 1.0),
    (2.0, 1.5),
    (3.0, 2.0),
    (4.0, 3.0),
    (1.0, 0.5)
]

T_total = 15.0  # final time long enough to see ~2 orbits


# figure 1: explicit Euler vs implicit Euler for multiple time steps
dt_values = [0.01, 0.05, 0.10]

for dt in dt_values:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(ICs)))

    for (x0, y0), col in zip(ICs, colors):
        # explicit Euler
        _, traj_e = explicit_euler(x0, y0, dt, T_total, params_c)
        axes[0].plot(
            traj_e[:, 0], traj_e[:, 1],
            color=col, lw=1.2,
            label=f"IC=({x0:.1f}, {y0:.1f})"
        )
        axes[0].plot(x0, y0, "o", color=col, markersize=5)
        add_arrows(axes[0], traj_e[:, 0], traj_e[:, 1], n_arrows=3, color=col)

        # implicit Euler
        _, traj_i, _ = implicit_euler(x0, y0, dt, T_total, params_c)
        axes[1].plot(
            traj_i[:, 0], traj_i[:, 1],
            color=col, lw=1.2,
            label=f"IC=({x0:.1f}, {y0:.1f})"
        )
        axes[1].plot(x0, y0, "o", color=col, markersize=5)
        add_arrows(axes[1], traj_i[:, 0], traj_i[:, 1], n_arrows=3, color=col)

    # mark equilibrium
    for ax in axes:
        ax.plot(1.0, 1.0, "k*", markersize=12, label="equilibrium (1,1)")
        style_phase_axes(ax)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)

    axes[0].set_title(rf"Explicit Euler, $\Delta t = {dt}$")
    axes[1].set_title(rf"Implicit Euler, $\Delta t = {dt}$")

    plt.suptitle(
        rf"Phase-space trajectories at $\Delta t = {dt}$ "
        r"($a=b=c=d=1$, $p=q=0$)"
    )
    plt.tight_layout()

    filename = f"fig_phase_dt_{dt:.2f}.png".replace(".", "p")
    plt.savefig(os.path.join(OUT, filename), dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[saved] {filename}")


# figure 2: effect of time step on explicit Euler (one IC, several dt)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
dt_list = [0.1, 0.05, 0.01]
ic_demo = (2.0, 1.0)

for ax, dt in zip(axes, dt_list):
    _, traj = explicit_euler(*ic_demo, dt, T_total, params_c)
    ax.plot(traj[:, 0], traj[:, 1], "C0-", lw=1.0)
    ax.plot(*ic_demo, "go", markersize=7, label="start")
    ax.plot(1.0, 1.0, "k*", markersize=12, label="equilibrium")
    add_arrows(ax, traj[:, 0], traj[:, 1], n_arrows=4, color="C0")
    ax.set_title(rf"Explicit Euler, $\Delta t = {dt}$")
    style_phase_axes(ax)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

plt.suptitle(r"Explicit Euler: outward spiral grows faster as $\Delta t$ "
             r"increases (IC = (2.0, 1.0))")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_explicit_dt_sweep.png"), dpi=180,
            bbox_inches="tight")
plt.close()
print("[saved] fig_explicit_dt_sweep.png")


# figure 3: effect of time step on IMPLICIT Euler (one IC, several dt)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, dt in zip(axes, dt_list):
    _, traj, _ = implicit_euler(*ic_demo, dt, T_total, params_c)
    ax.plot(traj[:, 0], traj[:, 1], "C3-", lw=1.0)
    ax.plot(*ic_demo, "go", markersize=7, label="start")
    ax.plot(1.0, 1.0, "k*", markersize=12, label="equilibrium")
    add_arrows(ax, traj[:, 0], traj[:, 1], n_arrows=4, color="C3")
    ax.set_title(rf"Implicit Euler, $\Delta t = {dt}$")
    style_phase_axes(ax)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

plt.suptitle(r"Implicit Euler: inward spiral shrinks faster as $\Delta t$ "
             r"increases (IC = (2.0, 1.0))")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_implicit_dt_sweep.png"), dpi=180,
            bbox_inches="tight")
plt.close()
print("[saved] fig_implicit_dt_sweep.png")


# figure 4: amplitude (radial distance from equilibrium) over time
# demonstrates the stability picture from part (d)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

T_long = 30.0
dt_a = 0.05
ic_a = (2.0, 1.0)

t_e, traj_e = explicit_euler(*ic_a, dt_a, T_long, params_c)
t_i, traj_i, _ = implicit_euler(*ic_a, dt_a, T_long, params_c)

r_e = np.sqrt((traj_e[:, 0] - 1.0) ** 2 + (traj_e[:, 1] - 1.0) ** 2)
r_i = np.sqrt((traj_i[:, 0] - 1.0) ** 2 + (traj_i[:, 1] - 1.0) ** 2)

axes[0].plot(t_e, r_e, "C0-", label="explicit Euler")
axes[0].plot(t_i, r_i, "C3-", label="implicit Euler")
axes[0].axhline(r_e[0], color="k", ls="--", lw=0.8,
                label="initial distance")
axes[0].set_xlabel("time $t$")
axes[0].set_ylabel(r"$\| u - u^* \|_2$")
axes[0].set_title(rf"Distance from equilibrium ($\Delta t = {dt_a}$)")
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8)

# theoretical amplification factor per step from linearisation (eigenvalues +/- i)
dts = np.linspace(0.001, 0.5, 200)
g_exp = np.sqrt(1.0 + dts ** 2)         # |1 + i*dt|
g_imp = 1.0 / np.sqrt(1.0 + dts ** 2)   # |1/(1 - i*dt)|

axes[1].plot(dts, g_exp, "C0-", label=r"explicit $|1 + i\Delta t|$")
axes[1].plot(dts, g_imp, "C3-", label=r"implicit $|1/(1 - i\Delta t)|$")
axes[1].axhline(1.0, color="k", ls="--", lw=0.8, label="stability boundary")
axes[1].set_xlabel(r"$\Delta t$")
axes[1].set_ylabel("per-step amplification $|G|$")
axes[1].set_title("Linearised amplification factor at (1,1)")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8)
axes[1].set_ylim(0.5, 1.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_stability.png"), dpi=180,
            bbox_inches="tight")
plt.close()
print("[saved] fig_stability.png")


# figure 5: large dt drives explicit Euler to genuine blow-up
fig, ax = plt.subplots(figsize=(6.5, 5))

dt_big = 0.4
T_big = 12.0
t_b, traj_b = explicit_euler(*ic_a, dt_big, T_big, params_c)
ax.plot(traj_b[:, 0], traj_b[:, 1], "C0o-", ms=3, lw=0.8)
ax.plot(*ic_a, "go", ms=7, label="start")
ax.plot(1.0, 1.0, "k*", ms=12, label="equilibrium")
add_arrows(ax, traj_b[:, 0], traj_b[:, 1], n_arrows=5, color="C0")

ax.set_title(rf"Explicit Euler with $\Delta t = {dt_big}$: "
             "trajectory leaves the physical region")
style_phase_axes(ax)
ax.legend(fontsize=8, loc="upper right")
ax.set_aspect("auto")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_explicit_blowup.png"), dpi=180,
            bbox_inches="tight")
plt.close()
print("[saved] fig_explicit_blowup.png")