"""Neighbor-density floater removal - a cleanup pass distinct from
`compress`'s pruning (which targets file size, not visual artifacts).

Clean-GS (smlab-niser/clean-gs, arXiv:2601.00913) is the closest published
match: whitelist filtering, depth-buffered color validation against rendered
views, and neighbor-density outlier removal. This implements only the last
part - the part that needs no rendered views and works directly on
`GaussianCloud.means`. The depth-buffered color-validation half is deferred
(see #66) since it needs known camera poses to be meaningfully accurate.
"""

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from splat.domain.gaussians import GaussianCloud


def floater_mask(points: np.ndarray, *, k: int = 16, std_ratio: float = 2.0) -> np.ndarray:
    """True for points that are *not* isolated floaters.

    For each point, estimate local density as the mean distance to its `k`
    nearest neighbors; a point is a floater if that distance exceeds the
    global mean by more than `std_ratio` standard deviations. Mirrors the
    standard statistical-outlier-removal criterion (PCL/Open3D's
    `remove_statistical_outlier`), applied here to Gaussian means.
    """
    n = points.shape[0]
    if n <= k:
        return np.ones(n, dtype=bool)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k + 1)
    mean_dist = distances[:, 1:].mean(axis=1)
    global_mean, global_std = mean_dist.mean(), mean_dist.std()
    if global_std == 0:
        return np.ones(n, dtype=bool)
    return mean_dist <= global_mean + std_ratio * global_std


class DensityDeclutterer:
    name = "density"

    def declutter(
        self, cloud: GaussianCloud, *, k: int = 16, std_ratio: float = 2.0, **params
    ) -> GaussianCloud:
        keep = floater_mask(cloud.means, k=k, std_ratio=std_ratio)
        sh_rest = cloud.sh_rest[keep] if cloud.sh_rest is not None else None
        return GaussianCloud(
            means=cloud.means[keep],
            scales=cloud.scales[keep],
            rotations=cloud.rotations[keep],
            opacities=cloud.opacities[keep],
            sh_dc=cloud.sh_dc[keep],
            sh_rest=sh_rest,
            sh_degree=cloud.sh_degree,
            scale_activation=cloud.scale_activation,
            opacity_activation=cloud.opacity_activation,
            metadata=replace(cloud.metadata),
        )
