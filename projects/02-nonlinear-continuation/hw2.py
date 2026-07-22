import numpy as np
from newton_solver import sine_mode_guess
from continuation import trace_multiple_branches
from plotting import plot_branch_norms, save_representative_contours, add_trivial_branch


def main():
    n_interior = 28  # number of interior points
    lam_11 = np.pi**2 * (1**2 + 1**2)  # 2pi^2
    lam_12 = np.pi**2 * (1**2 + 2**2)  # 5pi^2
    lam_21 = np.pi**2 * (2**2 + 1**2)  # 5pi^2

    # specifications for the different branches
    branch_specs = [
        {
            "name": "branch_11_pos",
            "u0": sine_mode_guess(n_interior, amplitude=+0.5, mode_x=1, mode_y=1),
            "lam0": lam_11 - 0.25,
            "dlam_init": -0.10,
            "ds": 0.10,
            "mode_x": 1,  # spatial pattern: 1 bump in x
            "mode_y": 1,  # spatial pattern: 1 bump in y
        },  # bowl / hill
        {
            "name": "branch_11_neg",
            "u0": sine_mode_guess(n_interior, amplitude=-0.5, mode_x=1, mode_y=1),
            "lam0": lam_11 + 0.25,
            "dlam_init": +0.10,
            "ds": 0.10,
            "mode_x": 1,  # spatial pattern: 1 bump in x
            "mode_y": 1,  # spatial pattern: 1 bump in y
        },  # bowl / hill
        {
            "name": "branch_12_pos",
            "u0": sine_mode_guess(n_interior, amplitude=+0.6, mode_x=1, mode_y=2),
            "lam0": lam_12 + 0.25,
            "dlam_init": +0.08,
            "ds": 0.08,
            "mode_x": 1,  # spatial pattern: 1 bump in x
            "mode_y": 2,  # spatial pattern: 2 bumps in y
        },  # 2 lobes vertically stacked
        {
            "name": "branch_12_neg",
            "u0": sine_mode_guess(n_interior, amplitude=-0.6, mode_x=1, mode_y=2),
            "lam0": lam_12 + 0.25,
            "dlam_init": +0.08,
            "ds": 0.08,
            "mode_x": 1,  # spatial pattern: 1 bump in x
            "mode_y": 2,  # spatial pattern: 2 bumps in y
        },  # 2 lobes vertically stacked
        {
            "name": "branch_21_pos",
            "u0": sine_mode_guess(n_interior, amplitude=+0.8, mode_x=2, mode_y=1),
            "lam0": lam_21 + 0.25,
            "dlam_init": +0.05,
            "ds": 0.05,
            "mode_x": 2,  # spatial pattern: 2 bumps in x
            "mode_y": 1,  # spatial pattern: 1 bump in y
        },  # 2 lobes horizontally stacked
        {
            "name": "branch_21_neg",
            "u0": sine_mode_guess(n_interior, amplitude=-0.8, mode_x=2, mode_y=1),
            "lam0": lam_21 + 0.25,
            "dlam_init": +0.05,
            "ds": 0.05,
            "mode_x": 2,  # spatial pattern: 2 bumps in x
            "mode_y": 1,  # spatial pattern: 1 bump in y
        }  # 2 lobes horizontally stacked
    ]

    # computes all solution branches using ALC
    branches = trace_multiple_branches(
        branch_specs=branch_specs,
        n_interior=n_interior,
        default_dlam_init=0.1,
        default_ds=0.1,
        n_steps=160,
        lam_min=0,
        lam_max=60,
        tol=1e-10,
        max_newton_iter=50,
        max_corr_iter=30,
        verbose=True,
    )

    # adds the trivial branch to the solution branches
    branches = add_trivial_branch(branches, lam_min=0, lam_max=60, npts=300)

    # plots the ||u||_2 (lam) vs lam
    plot_branch_norms(branches, outpath="results/branch_plot.png", use_signed=True)

    # saves samples of 3 representative contours to a directory
    save_representative_contours(branches, n_interior=n_interior, outdir="results/contours", n_samples=3)


if __name__ == "__main__":
    main()