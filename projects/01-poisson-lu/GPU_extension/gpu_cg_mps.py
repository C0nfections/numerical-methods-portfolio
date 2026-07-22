"""
Matrix-free Conjugate Gradient solver for the 2D Poisson equation
using the Apple GPU through PyTorch's MPS backend.

Problem:
    -∇²T = alpha       on [0, 1] x [0, 1]
    T = 0              on the boundary

The implementation stores only the interior grid values. It does not
construct the dense or sparse Poisson matrix.

Run:
    python gpu_cg_mps.py --nodes 64
    python gpu_cg_mps.py --nodes 512
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MPSCGStats:
    """Convergence and timing information for one CG solve."""

    nodes: int
    unknowns: int
    iterations: int
    converged: bool
    solve_seconds: float
    end_to_end_seconds: float
    absolute_residual: float
    relative_residual: float


def mps_available() -> bool:
    """Return True when PyTorch can access an Apple MPS GPU."""

    return (
        torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def require_mps() -> None:
    """Raise an informative error when MPS cannot be used."""

    if not torch.backends.mps.is_built():
        raise RuntimeError(
            "This PyTorch installation was not built with MPS support."
        )

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available. Confirm that you are using an "
            "Apple-silicon Mac and an ARM-native PyTorch installation."
        )


def apply_poisson(u: torch.Tensor) -> torch.Tensor:
    """
    Apply the scaled positive-definite five-point Poisson operator.

    This computes:

        L(u)[i,j] =
            4*u[i,j]
            - u[i-1,j]
            - u[i+1,j]
            - u[i,j-1]
            - u[i,j+1]

    The original discretized equation is:

        L(u) / h² = alpha.

    Therefore, this implementation solves the equivalent scaled system:

        L(u) = alpha * h².

    Avoiding division by h² keeps the tensor values at more moderate
    magnitudes, which is helpful when using float32 arithmetic.

    Zero padding enforces homogeneous Dirichlet boundary conditions.

    Parameters
    ----------
    u:
        Interior solution values with shape
        (n_interior, n_interior).

    Returns
    -------
    torch.Tensor
        The discrete Poisson operator applied to u.
    """

    if u.ndim != 2:
        raise ValueError(
            f"u must be a 2D tensor, but received shape {tuple(u.shape)}"
        )

    padded = F.pad(
        u,
        pad=(1, 1, 1, 1),
        mode="constant",
        value=0.0,
    )

    center = padded[1:-1, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    down = padded[:-2, 1:-1]
    up = padded[2:, 1:-1]

    return (
        4.0 * center
        - left
        - right
        - down
        - up
    )


def tensor_norm(x: torch.Tensor) -> torch.Tensor:
    """Compute the Euclidean norm of a tensor."""

    return torch.sqrt(torch.sum(x * x))


def warm_up_mps() -> None:
    """
    Initialize MPS before collecting benchmark timings.

    The first MPS operation includes runtime initialization overhead.
    """

    require_mps()

    device = torch.device("mps")

    with torch.no_grad():
        u = torch.ones(
            (32, 32),
            dtype=torch.float32,
            device=device,
        )

        result = apply_poisson(u)
        _ = torch.sum(result * result)

        torch.mps.synchronize()


def solve_poisson_mps(
    nodes: int,
    alpha: float = 2.0,
    rtol: float = 2e-4,
    atol: float = 0.0,
    maxiter: int = 20_000,
    check_every: int = 25,
) -> tuple[np.ndarray, MPSCGStats]:
    """
    Solve the 2D Poisson equation using matrix-free CG on MPS.

    Parameters
    ----------
    nodes:
        Total grid nodes per dimension, including boundary nodes.

        For example, nodes=64 produces a 62 x 62 interior grid.

    alpha:
        Constant source term in -∇²T = alpha.

    rtol:
        Relative residual tolerance.

        MPS uses float32 for this implementation. A tolerance around
        1e-4 to 2e-4 is more realistic than 1e-6 for this unpreconditioned
        PDE solve.

    atol:
        Absolute residual tolerance.

    maxiter:
        Maximum number of CG iterations.

    check_every:
        Number of CG iterations between true-residual checks.

        Computing the true residual and reading a scalar on the CPU
        requires GPU synchronization. Checking less frequently avoids
        synchronizing on every iteration.

    Returns
    -------
    full_grid:
        NumPy array with shape (nodes, nodes). Boundary values are zero.

    stats:
        Timing and convergence information.
    """

    require_mps()

    if nodes < 3:
        raise ValueError("nodes must be at least 3")

    if rtol <= 0.0:
        raise ValueError("rtol must be positive")

    if atol < 0.0:
        raise ValueError("atol cannot be negative")

    if maxiter < 1:
        raise ValueError("maxiter must be at least 1")

    if check_every < 1:
        raise ValueError("check_every must be at least 1")

    n_interior = nodes - 2
    unknowns = n_interior * n_interior
    h = 1.0 / (nodes - 1)

    device = torch.device("mps")
    dtype = torch.float32

    total_start = perf_counter()

    with torch.no_grad():
        # We solve the scaled system:
        #
        #     L(u) = alpha * h²
        #
        # where L is the dimensionless five-point operator.
        b = torch.full(
            (n_interior, n_interior),
            fill_value=float(alpha * h * h),
            dtype=dtype,
            device=device,
        )

        # Initial guess.
        x = torch.zeros_like(b)

        # Initial true residual.
        r = b - apply_poisson(x)
        p = r.clone()

        residual_squared = torch.sum(r * r)
        b_norm_tensor = tensor_norm(b)

        # Scalar setup values are read before solve timing begins.
        b_norm = float(b_norm_tensor.item())
        residual_target = max(
            float(atol),
            float(rtol) * b_norm,
        )
        residual_target_squared = residual_target * residual_target

        initial_residual_squared = float(
            residual_squared.item()
        )

        converged = (
            initial_residual_squared
            <= residual_target_squared
        )

        iterations = 0

        # Finish setup before beginning the solve-only timer.
        torch.mps.synchronize()
        solve_start = perf_counter()

        if not converged:
            for iteration in range(1, maxiter + 1):
                Ap = apply_poisson(p)

                denominator = torch.sum(p * Ap)
                step_size = residual_squared / denominator

                x = x + step_size * p
                r = r - step_size * Ap

                new_residual_squared = torch.sum(r * r)
                iterations = iteration

                should_check = (
                    iteration % check_every == 0
                    or iteration == maxiter
                )

                if should_check:
                    # The recursively updated CG residual can drift in
                    # float32. Recompute the actual residual from x.
                    true_residual = b - apply_poisson(x)
                    true_residual_squared = torch.sum(
                        true_residual * true_residual
                    )

                    true_residual_squared_value = float(
                        true_residual_squared.item()
                    )

                    if not np.isfinite(
                        true_residual_squared_value
                    ):
                        raise FloatingPointError(
                            "The true CG residual became NaN or infinite."
                        )

                    if (
                        true_residual_squared_value
                        <= residual_target_squared
                    ):
                        r = true_residual
                        residual_squared = (
                            true_residual_squared
                        )
                        converged = True
                        break

                    # Residual replacement:
                    #
                    # Use the independently computed residual instead
                    # of the drifted recursive residual. Restarting p
                    # sacrifices some conjugacy but is more robust in
                    # float32.
                    r = true_residual
                    p = true_residual.clone()
                    residual_squared = (
                        true_residual_squared
                    )

                    continue

                beta = (
                    new_residual_squared
                    / residual_squared
                )

                p = r + beta * p
                residual_squared = new_residual_squared

        # Ensure all iterations have completed before stopping the timer.
        torch.mps.synchronize()
        solve_seconds = perf_counter() - solve_start

        # Independently recompute the final residual. This value is the
        # source of truth for reporting convergence.
        final_residual = b - apply_poisson(x)
        final_absolute_residual_tensor = tensor_norm(
            final_residual
        )

        torch.mps.synchronize()

        absolute_residual = float(
            final_absolute_residual_tensor.item()
        )

        if b_norm > 0.0:
            relative_residual = (
                absolute_residual / b_norm
            )
        else:
            relative_residual = absolute_residual

        # Do not trust only the recursive CG stopping condition.
        # The final true residual determines convergence.
        converged = (
            absolute_residual
            <= residual_target
        )

        # Transfer the final interior solution to the CPU.
        interior_cpu = x.detach().cpu().numpy()

    full_grid = np.zeros(
        (nodes, nodes),
        dtype=np.float32,
    )

    full_grid[1:-1, 1:-1] = interior_cpu

    end_to_end_seconds = perf_counter() - total_start

    stats = MPSCGStats(
        nodes=nodes,
        unknowns=unknowns,
        iterations=iterations,
        converged=converged,
        solve_seconds=solve_seconds,
        end_to_end_seconds=end_to_end_seconds,
        absolute_residual=absolute_residual,
        relative_residual=relative_residual,
    )

    return full_grid, stats


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Solve the 2D Poisson equation using matrix-free "
            "Conjugate Gradient on an Apple MPS GPU."
        )
    )

    parser.add_argument(
        "--nodes",
        type=int,
        default=64,
        help=(
            "Total nodes per dimension, including boundaries "
            "(default: 64)."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Constant source term (default: 2.0).",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=2e-4,
        help=(
            "Relative residual tolerance "
            "(default: 2e-4)."
        ),
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help=(
            "Absolute residual tolerance "
            "(default: 0.0)."
        ),
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=20_000,
        help=(
            "Maximum CG iterations "
            "(default: 20000)."
        ),
    )

    parser.add_argument(
        "--check-every",
        type=int,
        default=25,
        help=(
            "Recompute the true residual every N iterations "
            "(default: 25)."
        ),
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip MPS warm-up before solving.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the MPS solver from the command line."""

    args = parse_arguments()

    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())

    if not args.skip_warmup:
        print("Warming up the MPS device...")
        warm_up_mps()

    grid, stats = solve_poisson_mps(
        nodes=args.nodes,
        alpha=args.alpha,
        rtol=args.rtol,
        atol=args.atol,
        maxiter=args.maxiter,
        check_every=args.check_every,
    )

    print()
    print("MPS Conjugate Gradient results")
    print("--------------------------------")
    print(f"Nodes per dimension: {stats.nodes}")
    print(f"Interior unknowns:   {stats.unknowns}")
    print(f"Iterations:          {stats.iterations}")
    print(f"Converged:           {stats.converged}")
    print(f"GPU solve time:      {stats.solve_seconds:.6f} s")
    print(
        f"End-to-end time:     "
        f"{stats.end_to_end_seconds:.6f} s"
    )
    print(
        f"Absolute residual:   "
        f"{stats.absolute_residual:.6e}"
    )
    print(
        f"Relative residual:   "
        f"{stats.relative_residual:.6e}"
    )
    print(f"Minimum solution:    {grid.min():.8f}")
    print(f"Maximum solution:    {grid.max():.8f}")

    if not stats.converged:
        raise SystemExit(
            "CG reached maxiter without satisfying the "
            "true-residual tolerance."
        )


if __name__ == "__main__":
    main()