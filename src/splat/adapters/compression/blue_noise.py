"""Weighted Sample Elimination (Yuksel, "Sample Elimination for Generating
Poisson Disk Sample Sets", 2015) - CPU-only blue-noise pruning that reduces a
dense point set to a target count while maximizing spatial uniformity, unlike
`prune_quantize`'s plain opacity/outlier thresholds, which don't consider
local density at all and can leave dense clusters untouched while thinning
sparse regions no differently.

`salivian/pixelpie` (GPU-rasterization-accelerated maximal Poisson-disk
sampling) was investigated as an inspiration but doesn't fit this problem
shape: it generates a maximal point set from scratch rather than reducing an
existing dense one to a target budget, and its acceleration is CUDA-only.
Weighted Sample Elimination solves the actual problem here and is CPU-only.

Implementation notes (deliberate deviations from the paper, not oversights):
  - The paper picks one global influence radius `r_max` from the domain
    volume and target count (an expected Poisson-disk packing radius). The
    domain volume is estimated from a 1st-99th percentile bounding box, not
    raw min/max - a real reconstruction's raw extent is easily dominated by
    a handful of far-flung SfM outlier points (the same concern
    `normalize_gaussian_cloud` already guards against via median, not mean).
  - Real Gaussian splat clouds are extremely non-uniform in density (surface-
    concentrated, not spread through their bounding volume), so a single
    global `r_max` gives some points astronomically more neighbors than
    others - candidate-pair counts on a real ~38k-point capture hit 10M+
    before this cap was added, which made the elimination loop impractically
    slow. `k_cap` bounds every point to its `k_cap` nearest candidates within
    `r_max`, capping total work at O(n * k_cap) regardless of local density.
  - `importance_bias` is this module's own addition, not from the paper:
    the paper's weight function is purely geometric. Every pairwise weight
    is scaled by its point's own `bias` (>1 for low-importance points), so a
    low-opacity Gaussian's elimination priority is inflated relative to an
    equally-crowded high-opacity one - "prefer eliminating low-contribution
    Gaussians when something in a crowded region has to go."
"""

import heapq

import numpy as np


def weighted_sample_elimination(
    points: np.ndarray,
    importance: np.ndarray,
    target_count: int,
    *,
    alpha: float = 8.0,
    beta: float = 2.0,
    k_cap: int = 16,
    importance_bias: float = 1.0,
) -> np.ndarray:
    """Returns a boolean keep-mask of length len(points), True for the
    `target_count` points selected.

    `importance` must be in [0, 1] (e.g. activated opacity) - higher
    importance is preferentially kept. `alpha` controls how sharply the
    per-pair weight falls off with distance (paper's tunable exponent).
    `beta` scales the estimated Poisson-disk radius into the elimination
    radius (paper's tunable multiplier - larger considers more neighbors
    per point, at higher cost). `k_cap` bounds how many of those neighbors
    each point actually keeps, for tractable runtime on non-uniform-density
    point clouds. `importance_bias` of 0 recovers the paper's purely-
    geometric elimination; higher values weight low-importance points more
    heavily for removal.
    """
    n = points.shape[0]
    if target_count >= n:
        return np.ones(n, dtype=bool)
    if target_count <= 0:
        return np.zeros(n, dtype=bool)

    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)
    bbox_volume = max(float(np.prod(hi - lo)), 1e-12)
    # expected packing radius for `target_count` spheres in this bounding volume
    r_low = (bbox_volume / (target_count * (4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)
    r_max = beta * r_low
    if r_max <= 0:
        return np.ones(n, dtype=bool)  # degenerate: all points coincide

    pi, pj, d = _candidate_pairs(points, r_max, k_cap)

    bias = 1.0 + importance_bias * (1.0 - np.clip(importance, 0.0, 1.0))
    weight = np.zeros(n, dtype=np.float64)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]

    if pi.size:
        kernel = np.clip(1.0 - d / r_max, 0.0, None) ** alpha
        keep_edge = kernel > 0.0
        for a, b, k in zip(
            pi[keep_edge].tolist(), pj[keep_edge].tolist(), kernel[keep_edge].tolist(), strict=True
        ):
            weight[a] += bias[a] * k
            weight[b] += bias[b] * k
            adjacency[a].append((b, k))
            adjacency[b].append((a, k))

    alive = np.ones(n, dtype=bool)
    heap = [(-weight[i], i) for i in range(n)]
    heapq.heapify(heap)

    remaining = n
    while remaining > target_count and heap:
        neg_w, i = heapq.heappop(heap)
        if not alive[i] or weight[i] != -neg_w:
            continue  # stale entry - its weight changed since this was pushed
        alive[i] = False
        remaining -= 1
        for j, k in adjacency[i]:
            if alive[j]:
                weight[j] -= bias[j] * k
                heapq.heappush(heap, (-weight[j], j))
    return alive


def _candidate_pairs(
    points: np.ndarray, r_max: float, k_cap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique (i, j) pairs within `r_max`, each point contributing at most
    its `k_cap` nearest candidates - bounds total pairs to O(n * k_cap)
    regardless of how non-uniform the point density is, instead of a plain
    radius query, which is unbounded in dense regions."""
    n = points.shape[0]
    k = min(k_cap, n - 1)
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    distances, neighbor_idx = tree.query(points, k=k + 1, distance_upper_bound=r_max)
    distances, neighbor_idx = distances[:, 1:], neighbor_idx[:, 1:]  # column 0 is self

    valid = neighbor_idx < n
    row_idx = np.repeat(np.arange(n), k)[valid.ravel()]
    col_idx = neighbor_idx.ravel()[valid.ravel()]
    dist = distances.ravel()[valid.ravel()]
    if row_idx.size == 0:
        return row_idx, col_idx, dist

    lo_idx = np.minimum(row_idx, col_idx)
    hi_idx = np.maximum(row_idx, col_idx)
    pair_keys = lo_idx.astype(np.int64) * n + hi_idx.astype(np.int64)
    _, unique_pos = np.unique(pair_keys, return_index=True)
    return lo_idx[unique_pos], hi_idx[unique_pos], dist[unique_pos]
