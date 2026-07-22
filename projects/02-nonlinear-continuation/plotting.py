import os
import numpy as np
import matplotlib.pyplot as plt

from newton_solver import vec_to_grid


def ensure_dir(path):
    # makes the directory
    os.makedirs(path, exist_ok=True)


def plot_branch_norms(branches, outpath="results/branch_plot.png", use_signed=True):
    # plot ||u||_2 vs lambda for all branches
    ensure_dir(os.path.dirname(outpath) or ".")
    plt.figure(figsize=(8, 6))

    for name, branch in branches.items():
        y = branch["signed_norms"] if use_signed else branch["norms"]
        plt.plot(branch["lams"], y, marker="o", markersize=3, linewidth=1.5, label=name)

    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"signed $\|u\|_2$" if use_signed else r"$\|u\|_2$")
    plt.title(r"Solution branches: $\|u\|_2(\lambda)$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-15, 20)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def add_trivial_branch(branches, lam_min=0, lam_max=60, npts=200):
    lams = np.linspace(lam_min, lam_max, npts)
    branches["trivial branch"] = {
        "lams": lams,
        "norms": np.zeros_like(lams),
        "signed_norms": np.zeros_like(lams),
        "solutions": [],
        "infos": [],
    }
    return branches


def plot_solution_contour(u_vec, n_interior, title, outpath):
    # save one contour plot for one solution
    ensure_dir(os.path.dirname(outpath) or ".")
    U = vec_to_grid(u_vec, n_interior)

    x = np.linspace(0.0, 1.0, n_interior + 2)
    y = np.linspace(0.0, 1.0, n_interior + 2)
    X, Y = np.meshgrid(x, y, indexing="ij")

    plt.figure(figsize=(6, 5))
    cp = plt.contourf(X, Y, U, levels=25)
    plt.colorbar(cp)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def save_representative_contours(branches, n_interior, outdir="results/contours", n_samples=3):
    """
    saves representative contour maps for each branch and
    picks 3 samples roughly equally spaced solutions along each branch
    """
    ensure_dir(outdir)

    for name, branch in branches.items():
        sols = branch["solutions"]
        lams = branch["lams"]

        if len(sols) == 0:
            continue

        sample_ids = np.linspace(0, len(sols) - 1, min(n_samples, len(sols)), dtype=int)

        for k, idx in enumerate(sample_ids):
            lam = lams[idx]
            u_vec = sols[idx]
            outpath = os.path.join(outdir, f"{name}_sample_{k+1}_lam_{lam:.3f}.png")
            title = f"{name}: contour at lambda = {lam:.3f}"
            plot_solution_contour(u_vec, n_interior, title, outpath)