"""Benchmark CPU and GPU CG implementations."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from cpu_cg import solve_poisson_cpu
from gpu_cg_cuda import solve_poisson_gpu


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512, 1024],
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    output_directory = Path("results")
    output_directory.mkdir(exist_ok=True)

    # Warm up the CUDA context and sparse libraries.
    print("Warming up GPU...")
    solve_poisson_gpu(nodes=32)

    rows = []

    for nodes in args.nodes:
        cpu_times = []
        gpu_times = []
        gpu_end_to_end_times = []

        cpu_residual = None
        gpu_residual = None
        iterations = None
        gpu_name = None

        for repeat in range(args.repeats):
            _, cpu_stats = solve_poisson_cpu(nodes)
            _, gpu_stats = solve_poisson_gpu(nodes)

            cpu_times.append(cpu_stats.solve_seconds)
            gpu_times.append(gpu_stats.solve_seconds)
            gpu_end_to_end_times.append(
                gpu_stats.end_to_end_seconds
            )

            cpu_residual = cpu_stats.relative_residual
            gpu_residual = gpu_stats.relative_residual
            iterations = gpu_stats.iterations
            gpu_name = gpu_stats.gpu_name

            print(
                f"nodes={nodes:5d}, "
                f"repeat={repeat + 1}, "
                f"CPU={cpu_stats.solve_seconds:.6f}s, "
                f"GPU={gpu_stats.solve_seconds:.6f}s"
            )

        cpu_median = statistics.median(cpu_times)
        gpu_median = statistics.median(gpu_times)
        end_to_end_median = statistics.median(
            gpu_end_to_end_times
        )

        speedup = cpu_median / gpu_median

        rows.append(
            {
                "nodes_per_dimension": nodes,
                "unknowns": (nodes - 2) ** 2,
                "cpu_cg_seconds": cpu_median,
                "gpu_cg_seconds": gpu_median,
                "gpu_end_to_end_seconds": end_to_end_median,
                "gpu_speedup_solve_only": speedup,
                "iterations": iterations,
                "cpu_relative_residual": cpu_residual,
                "gpu_relative_residual": gpu_residual,
                "gpu_name": gpu_name,
            }
        )

    output_path = output_directory / "benchmark.csv"

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    main()