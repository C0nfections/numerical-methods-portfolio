"""CPU sparse Conjugate Gradient solver for the 2D Poisson equation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import diags, kronsum
from scipy.sparse.linalg import cg


@dataclass(frozen=True)
class SolverStats:
    solve_seconds: float
    iterations: int
    relative_residual: float


def build_poisson_system(
    nodes: int,
    alpha: float,
) -> tuple:
    """
    Construct the sparse system for

        -Laplacian(T) = alpha

    on [0, 1] x [0, 1] with T = 0 on the boundary.

    Parameters
    ----------
    nodes:
        Total nodes per spatial direction, including boundaries.
    alpha:
        Constant source term.
    """
    if nodes < 3:
        raise ValueError("nodes must be at least 3")

    n = nodes - 2
    h = 1.0 / (nodes - 1)

    off_diagonal = -np.ones(n - 1, dtype=np.float64)
    diagonal = 2.0 * np.ones(n, dtype=np.float64)

    one_dimensional = diags(
        diagonals=[off_diagonal, diagonal, off_diagonal],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="csr",
    )

    # A = kron(I, T) + kron(T, I)
    A = kronsum(
        one_dimensional,
        one_dimensional,
        format="csr",
    ) / (h * h)

    b = np.full(n * n, alpha, dtype=np.float64)

    return A, b


def solve_poisson_cpu(
    nodes: int,
    alpha: float = 2.0,
    rtol: float = 1e-8,
    maxiter: int = 10_000,
) -> tuple[np.ndarray, SolverStats]:
    A, b = build_poisson_system(nodes, alpha)

    iteration_count = 0

    def callback(_: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1

    start = perf_counter()

    solution, info = cg(
        A,
        b,
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
        callback=callback,
    )

    solve_seconds = perf_counter() - start

    if info != 0:
        raise RuntimeError(
            f"CPU CG did not converge; scipy returned info={info}"
        )

    residual = A @ solution - b
    relative_residual = (
        np.linalg.norm(residual) / np.linalg.norm(b)
    )

    grid = np.zeros((nodes, nodes), dtype=np.float64)
    grid[1:-1, 1:-1] = solution.reshape(nodes - 2, nodes - 2)

    stats = SolverStats(
        solve_seconds=solve_seconds,
        iterations=iteration_count,
        relative_residual=float(relative_residual),
    )

    return grid, stats


if __name__ == "__main__":
    result, stats = solve_poisson_cpu(nodes=128)

    print(f"CPU solve time: {stats.solve_seconds:.6f} seconds")
    print(f"Iterations: {stats.iterations}")
    print(f"Relative residual: {stats.relative_residual:.3e}")
    print(f"Maximum temperature: {result.max():.8f}")