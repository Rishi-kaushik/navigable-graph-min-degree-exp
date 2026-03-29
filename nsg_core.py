"""Core experiment logic for navigable-graph minimum-degree studies."""

import math
from typing import List, Optional

import numpy as np


if hasattr(int, "bit_count"):
    def popcount(x: int) -> int:
        """Return number of set bits in an integer mask."""
        return x.bit_count()  # type: ignore[attr-defined]
else:
    def popcount(x: int) -> int:
        """Return number of set bits in an integer mask."""
        return bin(x).count("1")


def generate_points(
    n: int,
    dim: int,
    distribution: str,
    normalize_points: bool,
    seed: int,
) -> np.ndarray:
    """Generate random points in R^dim with optional L2 normalization."""
    rng = np.random.default_rng(seed)

    if distribution == "uniform":
        points = rng.random((n, dim), dtype=np.float64)
    elif distribution == "gaussian":
        points = rng.standard_normal((n, dim), dtype=np.float64)
    else:
        raise ValueError("distribution must be 'uniform' or 'gaussian'")

    if normalize_points:
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Guard against divide-by-zero for zero vectors.
        np.maximum(norms, np.finfo(np.float64).tiny, out=norms)
        points = points / norms

    return points


def pairwise_squared_distances(points: np.ndarray, assume_unit_norm: bool) -> np.ndarray:
    """Build full squared Euclidean distance matrix.

    Uses the Gram-matrix identity:
      ||u - v||^2 = ||u||^2 + ||v||^2 - 2 * (u dot v)

    If points are L2-normalized (unit length), this simplifies to:
      ||u - v||^2 = 2 - 2 * (u dot v)
    """
    # (n, d) @ (d, n) -> (n, n)
    # Some BLAS backends may emit spurious overflow warnings; validate output.
    with np.errstate(over="ignore", invalid="ignore"):
        gram = points @ points.T
    if not np.isfinite(gram).all():
        raise FloatingPointError("non-finite Gram matrix; input magnitudes too large")
    if assume_unit_norm:
        dists = 2.0 - 2.0 * gram
    else:
        sq_norms = np.einsum("ij,ij->i", points, points, optimize=True)
        dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram

    # Numerical cleanup.
    dists = np.asarray(dists, dtype=np.float64)
    if not np.isfinite(dists).all():
        raise FloatingPointError("non-finite distance matrix; input magnitudes too large")
    np.maximum(dists, 0.0, out=dists)
    np.fill_diagonal(dists, 0.0)
    return dists


def greedy_cover_degree_for_source(
    source: int,
    dists: np.ndarray,
    cutoff_degree: Optional[int],
) -> int:
    """Greedy set-cover out-degree for one source with optional pruning cutoff."""
    n = len(dists)

    universe = 0
    for b in range(n):
        if b != source:
            universe |= 1 << b

    source_row = dists[source]
    candidates: List[int] = []
    for c in range(n):
        if c == source:
            continue
        c_row = dists[c]
        mask = 0
        for b in range(n):
            if b != source and c_row[b] < source_row[b]:
                mask |= 1 << b
        if mask:
            candidates.append(mask)

    uncovered = universe
    degree = 0
    while uncovered:
        best_gain = 0
        best_cover = 0
        for mask in candidates:
            cover = mask & uncovered
            gain = popcount(cover)
            if gain > best_gain:
                best_gain = gain
                best_cover = cover

        if best_gain == 0:
            # Numerical fallback: directly cover one remaining target.
            best_cover = uncovered & -uncovered

        uncovered &= ~best_cover
        degree += 1

        # If already worse than known min, this source cannot improve global min.
        if cutoff_degree is not None and degree > cutoff_degree:
            return degree

    return degree


def run_trial_min_only(
    n: int,
    dim: int,
    distribution: str,
    normalize_points: bool,
    seed: int,
) -> int:
    """Run one experiment and return minimum out-degree over all sources."""
    points = generate_points(n, dim, distribution, normalize_points, seed)
    dists = pairwise_squared_distances(points, assume_unit_norm=normalize_points)

    current_min = n - 1
    for source in range(n):
        deg = greedy_cover_degree_for_source(source, dists, cutoff_degree=current_min)
        if deg < current_min:
            current_min = deg
            if current_min <= 1:
                break
    return current_min
