"""Output helpers: run directory creation, CSV export, and heatmap plotting."""

import csv
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence

import matplotlib
import numpy as np

# Force headless backend for terminal/batch use.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# Some matplotlib/numpy internals may emit harmless divide/invalid runtime warnings
# on constant heatmaps; suppress these to avoid warning-as-error failures.
np.seterr(divide="ignore", invalid="ignore")
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=r"divide by zero encountered.*"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=r"invalid value encountered.*"
)


def create_run_dir(results_root: str) -> Path:
    """Create a timestamped run directory under results_root."""
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)

    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / base
    suffix = 1
    while run_dir.exists():
        run_dir = root / (base + "_" + str(suffix))
        suffix += 1

    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def write_run_config(path: Path, config: Dict[str, object]) -> None:
    """Write run configuration as JSON for reproducibility."""
    with path.open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def save_grid_csv(
    path: Path,
    dims: Sequence[int],
    point_counts: Sequence[int],
    avg_min_vals: Sequence[Sequence[float]],
    max_min_vals: Sequence[Sequence[float]],
    trials_used: Sequence[Sequence[int]],
) -> None:
    """Write average-min, max-min, and trials-used matrices to a CSV file."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        header = [str(n) for n in point_counts]

        w.writerow(["avg_min_degree"] + header)
        for i, dim in enumerate(dims):
            w.writerow([dim] + ["{:.6f}".format(v) for v in avg_min_vals[i]])

        w.writerow([])
        w.writerow(["max_min_degree"] + header)
        for i, dim in enumerate(dims):
            w.writerow([dim] + ["{:.6f}".format(v) for v in max_min_vals[i]])

        w.writerow([])
        w.writerow(["trials_used"] + header)
        for i, dim in enumerate(dims):
            w.writerow([dim] + [str(v) for v in trials_used[i]])


def plot_heatmap(
    path: Path,
    dims: Sequence[int],
    point_counts: Sequence[int],
    values: Sequence[Sequence[float]],
    title: str,
    cbar_label: str,
    annotate_cells: bool,
) -> None:
    """Render heatmap to PNG (or any matplotlib-supported image extension)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot plot empty heatmap")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    elif vmax <= vmin:
        # Avoid normalization divide-by-zero on constant heatmaps.
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(14, 6))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        im = ax.imshow(
            arr,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    x_idx = list(range(len(point_counts)))
    ax.set_xticks(x_idx)
    ax.set_xticklabels([str(point_counts[i]) for i in x_idx], rotation=90, ha="center", fontsize=7)

    y_idx = list(range(len(dims)))
    ax.set_yticks(y_idx)
    ax.set_yticklabels([str(d) for d in dims])

    if annotate_cells:
        rows = len(values)
        cols = len(values[0]) if rows > 0 else 0
        font_size = 4 if cols > 25 else 6
        for i in range(rows):
            for j in range(cols):
                v = values[i][j]
                label = ("{:.2f}".format(v)).rstrip("0").rstrip(".")
                rgba = im.cmap(im.norm(v))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "black" if luminance > 0.6 else "white"
                ax.text(j, i, label, ha="center", va="center", fontsize=font_size, color=text_color)

    ax.set_xlabel("Number of points (n)")
    ax.set_ylabel("Dimension (d)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(path), dpi=180)
    plt.close(fig)
