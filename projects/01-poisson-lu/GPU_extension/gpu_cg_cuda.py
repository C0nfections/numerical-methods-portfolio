"""GPU sparse Conjugate Gradient solver using CuPy."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.sparse import diags, kronsum
    from cupyx.scipy.sparse.linalg import cg
except ImportError as exc:
    raise ImportError(
        "CuPy is required for gpu_cg_cuda.py. "
        "Install the CuPy package matching your CUDA version."
    ) from exc


@dataclass(frozen=True)
class GPUSolverStats:
    solve_seconds: float
    end_to_end_seconds: float
    iterations: int
    relative_residual: float
    gpu_name: str


def gpu_available() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def build_gpu_poisson_system(
    nodes: int,
    alpha: float,
):
    if nodes < 3:
        raise ValueError("nodes must be at least 3")

    n = nodes - 2
    h = 1.0 / (nodes - 1)

    off_diagonal = -cp.ones(n - 1, dtype=cp.float64)
    diagonal = 2.0 * cp.ones(n, dtype=cp.float64)

    one_dimensional = diags(
        diagonals=[off_diagonal, diagonal, off_diagonal],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="csr",
    )

    A = kronsum(
        one_dimensional,
        one_dimensional,
        format="csr",
    ) / (h * h)

    b = cp.full(n * n, alpha, dtype=cp.float64)

    return A, b


def solve_poisson_gpu(
    nodes: int,
    alpha: float = 2.0,
    rtol: float = 1e-8,
    maxiter: int = 10_000,
) -> tuple[np.ndarray, GPUSolverStats]:
    if not gpu_available():
        raise RuntimeError("No CUDA-capable GPU was detected.")

    total_start = perf_counter()

    A, b = build_gpu_poisson_system(nodes, alpha)

    iteration_count = 0

    def callback(_) -> None:
        nonlocal iteration_count
        iteration_count += 1

    # CUDA work is asynchronous, so use CUDA events for solve timing.
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    start_event.record()

    solution, info = cg(
        A,
        b,
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
        callback=callback,
    )

    end_event.record()
    end_event.synchronize()

    solve_seconds = (
        cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
    )

    if info != 0:
        raise RuntimeError(
            f"GPU CG did not converge; CuPy returned info={info}"
        )

    residual = A @ solution - b
    relative_residual = (
        cp.linalg.norm(residual) / cp.linalg.norm(b)
    ).item()

    # Copy the result to the CPU after the solve timing has ended.
    solution_cpu = cp.asnumpy(solution)

    grid = np.zeros((nodes, nodes), dtype=np.float64)
    grid[1:-1, 1:-1] = solution_cpu.reshape(
        nodes - 2,
        nodes - 2,
    )

    cp.cuda.Stream.null.synchronize()
    end_to_end_seconds = perf_counter() - total_start

    properties = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = properties["name"].decode("utf-8")

    stats = GPUSolverStats(
        solve_seconds=solve_seconds,
        end_to_end_seconds=end_to_end_seconds,
        iterations=iteration_count,
        relative_residual=float(relative_residual),
        gpu_name=gpu_name,
    )

    return grid, stats


if __name__ == "__main__":
    # Warm-up run so initialization is excluded from the reported run.
    solve_poisson_gpu(nodes=32)

    result, stats = solve_poisson_gpu(nodes=512)

    print(f"GPU: {stats.gpu_name}")
    print(f"GPU solve time: {stats.solve_seconds:.6f} seconds")
    print(
        "End-to-end time: "
        f"{stats.end_to_end_seconds:.6f} seconds"
    )
    print(f"Iterations: {stats.iterations}")
    print(f"Relative residual: {stats.relative_residual:.3e}")
    print(f"Maximum temperature: {result.max():.8f}")