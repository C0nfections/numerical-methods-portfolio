"""Correctness tests comparing the CPU and Apple MPS Poisson solvers."""

import numpy as np
import pytest
import torch

from cpu_cg import solve_poisson_cpu
from gpu_cg_mps import warm_up_mps, solve_poisson_mps


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple MPS GPU is not available.",
)


@pytest.fixture(scope="module", autouse=True)
def warm_up_gpu() -> None:
    """Warm up MPS once before running this test module."""

    warm_up_mps()


@pytest.mark.parametrize("nodes", [16, 32, 64])
def test_mps_solution_matches_cpu(nodes: int) -> None:
    """Verify that CPU and MPS solvers produce similar solutions."""

    cpu_grid, cpu_stats = solve_poisson_cpu(
        nodes=nodes,
        alpha=2.0,
        rtol=1e-8,
    )

    gpu_grid, gpu_stats = solve_poisson_mps(
        nodes=nodes,
        alpha=2.0,
        rtol=2e-4,
    )

    relative_difference = (
        np.linalg.norm(cpu_grid - gpu_grid)
        / np.linalg.norm(cpu_grid)
    )

    print()
    print(f"Grid: {nodes} x {nodes}")
    print(
        "CPU residual:",
        cpu_stats.relative_residual,
    )
    print(
        "GPU residual:",
        gpu_stats.relative_residual,
    )
    print(
        "Relative CPU/GPU solution difference:",
        relative_difference,
    )

    assert gpu_stats.converged
    assert gpu_stats.relative_residual < 3e-4
    assert relative_difference < 1e-4


def test_mps_boundary_conditions() -> None:
    """Verify that all four boundaries remain zero."""

    grid, stats = solve_poisson_mps(
        nodes=64,
        alpha=2.0,
        rtol=2e-4,
    )

    assert stats.converged
    assert np.allclose(grid[0, :], 0.0)
    assert np.allclose(grid[-1, :], 0.0)
    assert np.allclose(grid[:, 0], 0.0)
    assert np.allclose(grid[:, -1], 0.0)


def test_mps_solution_is_finite_and_positive() -> None:
    """Check basic physical properties of the Poisson solution."""

    grid, stats = solve_poisson_mps(
        nodes=64,
        alpha=2.0,
        rtol=2e-4,
    )

    assert stats.converged
    assert np.isfinite(grid).all()
    assert grid.min() >= -1e-7
    assert grid.max() > 0.0