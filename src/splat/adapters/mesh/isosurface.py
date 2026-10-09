"""Level-set meshing of a Gaussian cloud: splat opacity into a voxel grid,
blur it to the Gaussians' footprint, and extract the surface with marching
cubes. Pure geometry, deterministic given its inputs: the `isosurface`
backend of `splat mesh`.
"""

import mcubes
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from splat.adapters.formats.splat_fmt import SH_C0
from splat.domain.errors import SplatDomainError
from splat.domain.gaussians import GaussianCloud
from splat.domain.image_space import Shape3D

MIN_POINTS = 4
TRIM = 1.0  # percentile of floaters ignored on each side when sizing the grid
LEVEL = 0.5  # surface density as a fraction of the median occupied voxel


def isosurface_mesh(
    cloud: GaussianCloud, *, resolution: int = 192, opacity_threshold: float = 0.1
) -> Shape3D:
    activated = cloud.to_activated_opacities()
    kept = np.flatnonzero(activated >= opacity_threshold)
    if len(kept) < MIN_POINTS:
        kept = np.arange(cloud.point_count)
    if len(kept) < MIN_POINTS:
        raise SplatDomainError(
            f"isosurface meshing needs at least {MIN_POINTS} points, got {len(kept)}"
        )
    sigma = float(np.median(cloud.to_linear_scales()[kept]))
    lo, hi = np.percentile(cloud.means[kept], [TRIM, 100 - TRIM], axis=0)
    kept = kept[((cloud.means[kept] >= lo) & (cloud.means[kept] <= hi)).all(axis=1)]
    means = cloud.means[kept].astype(np.float64)

    low = lo - 3 * sigma
    extent = hi + 3 * sigma - low
    voxel = float(extent.max()) / resolution
    shape = np.ceil(extent / voxel).astype(int) + 1
    flat = np.ravel_multi_index(np.floor((means - low) / voxel).astype(int).T, shape)
    field = np.bincount(flat, weights=activated[kept], minlength=shape.prod()).reshape(shape)
    field = gaussian_filter(field, sigma=max(sigma / voxel, 1.0))

    vertices, faces = mcubes.marching_cubes(
        field, LEVEL * np.median(field[field > 1e-3 * field.max()])
    )
    if len(faces) == 0:
        raise SplatDomainError("isosurface meshing found no surface; lower --opacity-threshold.")
    vertices = vertices * voxel + low

    colors = np.clip(0.5 + SH_C0 * cloud.sh_dc[kept], 0.0, 1.0)
    _, nearest = cKDTree(means).query(vertices)
    return Shape3D(
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int64),
        colors=(colors[nearest] * 255).astype(np.uint8),
        metadata={
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "resolution": resolution,
        },
    )
