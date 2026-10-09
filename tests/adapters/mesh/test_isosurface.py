import numpy as np
import pytest

from splat.adapters.mesh.isosurface import isosurface_mesh
from splat.domain.errors import SplatDomainError
from splat.domain.gaussians import GaussianCloud


def _sphere_cloud(n: int = 500, *, opacity: float = 10.0) -> GaussianCloud:
    # Fibonacci sphere sampling - a coherent surface, unlike a random cloud,
    # so the isosurface has real geometry to converge on.
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    golden = np.pi * (1 + 5**0.5)
    theta = golden * i
    means = np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1
    ).astype(np.float32)
    return GaussianCloud(
        means=means,
        scales=np.full((n, 3), -3.0, dtype=np.float32),
        rotations=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)),
        opacities=np.full(n, opacity, dtype=np.float32),
        sh_dc=np.full((n, 3), 0.2, dtype=np.float32),
        sh_degree=0,
    )


def test_reconstructs_a_surface_around_the_sphere():
    shape = isosurface_mesh(_sphere_cloud(), resolution=64)

    radii = np.linalg.norm(shape.vertices, axis=1)
    assert len(shape.faces) > 0
    assert abs(np.median(radii) - 1.0) < 0.1
    assert shape.metadata["resolution"] == 64


def test_colors_are_per_vertex():
    shape = isosurface_mesh(_sphere_cloud(), resolution=32)

    assert shape.colors.shape == (len(shape.vertices), 3)
    assert shape.colors.dtype == np.uint8


def test_higher_resolution_gives_more_faces():
    coarse = isosurface_mesh(_sphere_cloud(), resolution=32)
    fine = isosurface_mesh(_sphere_cloud(), resolution=96)

    assert len(fine.faces) > len(coarse.faces)


def test_rejects_too_few_points():
    with pytest.raises(SplatDomainError):
        isosurface_mesh(_sphere_cloud(n=3))


def test_falls_back_when_opacity_filter_leaves_too_few_points():
    cloud = _sphere_cloud(n=500, opacity=-10.0)  # activated opacity near 0 for every point

    assert len(isosurface_mesh(cloud, resolution=32, opacity_threshold=0.5).faces) > 0
