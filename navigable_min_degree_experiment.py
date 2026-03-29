#!/usr/bin/env python3
"""Run navigable-graph min-degree experiments and save timestamped results."""

import argparse
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from nsg_core import run_trial_min_only
from nsg_outputs import create_run_dir, plot_heatmap, save_grid_csv, write_run_config


def parse_args() -> argparse.Namespace:
    """Parse CLI options for grid-search mode."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Complete grid-search command (all options):\n"
            "  python3 navigable_min_degree_experiment.py \\\n"
            "    --distribution gaussian \\\n"
            "    --normalize-points \\\n"
            "    --seed 0 \\\n"
            "    --points-start 10 --points-end 200 --points-multiplier 2 \\\n"
            "    --dims-start 5 --dims-end 20 --dims-step 1 \\\n"
            "    --experiments-per-cell 10 \\\n"
            "    --workers 8 \\\n"
            "    --annotate-cells \\\n"
            "    --results-root results \\\n"
            "    --csv-name min_degree_heatmap.csv \\\n"
            "    --avg-heatmap-name avg_min_degree_heatmap.png \\\n"
            "    --max-heatmap-name max_min_degree_heatmap.png\n"
        ),
    )

    # Experiment options.
    p.add_argument("--distribution", choices=["uniform", "gaussian"], default="gaussian")
    p.add_argument("--normalize-points", action="store_true", help="L2-normalize each sampled point")
    p.add_argument("--seed", type=int, default=0)
 

    # Grid options.
    p.add_argument("--points-start", type=int, default=10)
    p.add_argument("--points-end", type=int, default=200)
    p.add_argument(
        "--points-step",
        type=int,
        default=5,
        help="Deprecated (ignored): n is generated exponentially",
    )
    p.add_argument(
        "--points-multiplier",
        type=float,
        default=2.0,
        help="Exponential growth factor for n (default: 2.0)",
    )
    p.add_argument("--dims-start", type=int, default=5)
    p.add_argument("--dims-end", type=int, default=20)
    p.add_argument("--dims-step", type=int, default=1)
    p.add_argument("--experiments-per-cell", type=int, default=10)
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1)),
        help="Number of worker processes for cell-level parallelism",
    )
    p.add_argument("--annotate-cells", action="store_true", help="Overlay values on heatmap cells")

    # Output options.
    p.add_argument("--results-root", type=str, default="results", help="Root folder for timestamped run directories")
    p.add_argument("--csv-name", type=str, default="min_degree_heatmap.csv")
    p.add_argument("--avg-heatmap-name", type=str, default="avg_min_degree_heatmap.png")
    p.add_argument("--max-heatmap-name", type=str, default="max_min_degree_heatmap.png")

    return p.parse_args()


CellTask = Tuple[int, int, int, int, Sequence[int], str, bool]
CellResult = Tuple[int, int, int, int, float, float, int]


def _run_cell_task(task: CellTask) -> CellResult:
    """Worker: run all experiments for one (dim, n) cell."""
    i, j, n, dim, seeds, distribution, normalize_points = task

    min_vals: List[int] = []
    for seed in seeds:
        min_vals.append(
            run_trial_min_only(
                n=n,
                dim=dim,
                distribution=distribution,
                normalize_points=normalize_points,
                seed=seed,
            )
        )

    avg_min = sum(min_vals) / float(len(min_vals))
    max_min = float(max(min_vals))
    return i, j, n, dim, avg_min, max_min, len(min_vals)


def _build_tasks(args: argparse.Namespace, dims: Sequence[int], point_counts: Sequence[int]) -> List[CellTask]:
    """Create deterministic per-cell/per-experiment seeds from base seed."""
    seed_rng = random.Random(args.seed)
    tasks: List[CellTask] = []

    for i, dim in enumerate(dims):
        for j, n in enumerate(point_counts):
            seeds = [seed_rng.getrandbits(63) for _ in range(args.experiments_per_cell)]
            tasks.append((i, j, n, dim, seeds, args.distribution, args.normalize_points))

    return tasks


def run_grid(args: argparse.Namespace, run_dir: Path) -> None:
    """Execute grid search, then save CSV + average/max heatmaps."""
    if args.experiments_per_cell <= 0:
        raise ValueError("--experiments-per-cell must be positive")
    if args.dims_step <= 0:
        raise ValueError("--dims-step must be positive")
    if args.points_start < 2:
        raise ValueError("--points-start must be at least 2")
    if args.points_end < args.points_start:
        raise ValueError("--points-end must be >= --points-start")
    if not (args.points_multiplier > 1.0):
        raise ValueError("--points-multiplier must be > 1")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    # Generate n values exponentially: start, start*m, start*m^2, ... <= end.
    point_counts: List[int] = []
    n = int(args.points_start)
    while n <= args.points_end:
        point_counts.append(n)
        next_n = int(math.ceil(float(n) * float(args.points_multiplier)))
        if next_n <= n:
            # Safety guard against float rounding or pathological multipliers.
            next_n = n + 1
        n = next_n

    dims = list(range(args.dims_start, args.dims_end + 1, args.dims_step))
    if not point_counts or not dims:
        raise ValueError("grid ranges must produce at least one n and one dim")

    avg_min_vals: List[List[float]] = [[0.0] * len(point_counts) for _ in dims]
    max_min_vals: List[List[float]] = [[0.0] * len(point_counts) for _ in dims]
    trials_used: List[List[int]] = [[0] * len(point_counts) for _ in dims]

    tasks = _build_tasks(args, dims, point_counts)
    total_cells = len(tasks)

    if args.workers == 1:
        for done, task in enumerate(tasks, start=1):
            i, j, n, dim, avg_min, max_min, used = _run_cell_task(task)
            avg_min_vals[i][j] = avg_min
            max_min_vals[i][j] = max_min
            trials_used[i][j] = used
            print(
                f"[{done}/{total_cells}] n={n}, dim={dim} -> "
                f"avg_min={avg_min:.3f}, max_min={int(max_min)}, trials_used={used}"
            )
    else:
        done = 0
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(_run_cell_task, task) for task in tasks]
                for fut in as_completed(futures):
                    done += 1
                    i, j, n, dim, avg_min, max_min, used = fut.result()
                    avg_min_vals[i][j] = avg_min
                    max_min_vals[i][j] = max_min
                    trials_used[i][j] = used
                    print(
                        f"[{done}/{total_cells}] n={n}, dim={dim} -> "
                        f"avg_min={avg_min:.3f}, max_min={int(max_min)}, trials_used={used}"
                    )
        except (PermissionError, OSError, RuntimeError) as exc:
            print(
                f"parallel execution unavailable ({exc}); falling back to sequential mode"
            )
            for done, task in enumerate(tasks, start=1):
                i, j, n, dim, avg_min, max_min, used = _run_cell_task(task)
                avg_min_vals[i][j] = avg_min
                max_min_vals[i][j] = max_min
                trials_used[i][j] = used
                print(
                    f"[{done}/{total_cells}] n={n}, dim={dim} -> "
                    f"avg_min={avg_min:.3f}, max_min={int(max_min)}, trials_used={used}"
                )

    csv_path = run_dir / Path(args.csv_name).name
    avg_png_path = run_dir / Path(args.avg_heatmap_name).name
    max_png_path = run_dir / Path(args.max_heatmap_name).name

    save_grid_csv(
        path=csv_path,
        dims=dims,
        point_counts=point_counts,
        avg_min_vals=avg_min_vals,
        max_min_vals=max_min_vals,
        trials_used=trials_used,
    )
    plot_heatmap(
        path=avg_png_path,
        dims=dims,
        point_counts=point_counts,
        values=avg_min_vals,
        title="Average Minimum Degree Heatmap",
        cbar_label="Average of per-experiment minimum degree",
        annotate_cells=args.annotate_cells,
    )
    plot_heatmap(
        path=max_png_path,
        dims=dims,
        point_counts=point_counts,
        values=max_min_vals,
        title="Maximum Minimum Degree Heatmap",
        cbar_label="Maximum of per-experiment minimum degree",
        annotate_cells=args.annotate_cells,
    )

    print("wrote csv:", csv_path)
    print("wrote avg heatmap:", avg_png_path)
    print("wrote max heatmap:", max_png_path)


def main() -> None:
    """Entry point: create run dir, save config, and execute grid run."""
    args = parse_args()
    run_dir = create_run_dir(args.results_root)

    config: Dict[str, object] = vars(args).copy()
    config["run_dir"] = str(run_dir)
    write_run_config(run_dir / "run_config.json", config)

    print("run directory:", run_dir)
    run_grid(args, run_dir)


if __name__ == "__main__":
    main()
