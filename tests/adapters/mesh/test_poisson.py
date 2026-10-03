import json

import numpy as np
import pytest

from splat.adapters.mesh.poisson import PoissonMeshExporter
from splat.domain.errors import SplatDomainError, UnsupportedFormat
from splat.domain.gaussians import GaussianCloud


def _sphere_cloud(n: int = 500, *, opacity: float = 10.0) -> GaussianCloud:
    # Fibonacci sphere sampling - a coherent surface, unlike a random cloud,
    # so Poisson reconstruction has real geometry to converge on.
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


def test_export_reconstructs_a_real_surface(tmp_path):
    cloud = _sphere_cloud()
    out_path = tmp_path / "sphere.obj"

    vertex_count, face_count = PoissonMeshExporter().export(cloud, out_path, format="obj", depth=6)

    assert out_path.exists()
    assert vertex_count > 0
    assert face_count > 0


def test_export_rejects_unsupported_format(tmp_path):
    cloud = _sphere_cloud()

    with pytest.raises(UnsupportedFormat):
        PoissonMeshExporter().export(cloud, tmp_path / "sphere.usdz", format="usdz")


def test_export_rejects_too_few_points(tmp_path):
    cloud = _sphere_cloud(n=3)

    with pytest.raises(SplatDomainError):
        PoissonMeshExporter().export(cloud, tmp_path / "sphere.obj", format="obj")


def test_export_falls_back_when_opacity_filter_leaves_too_few_points(tmp_path):
    cloud = _sphere_cloud(n=500, opacity=-10.0)  # activated opacity near 0 for every point
    out_path = tmp_path / "sphere.obj"

    vertex_count, face_count = PoissonMeshExporter().export(
        cloud, out_path, format="obj", depth=6, opacity_threshold=0.5
    )

    assert vertex_count > 0
    assert face_count > 0


def test_poisson_mesh_retries_lower_when_the_worker_exits_without_a_mesh(monkeypatch):
    """PoissonRecon exits its process with status 0 when isosurface extraction fails."""
    import subprocess

    from splat.adapters.mesh import poisson

    real_run = subprocess.run
    depths = []

    def flaky_run(cmd, **kwargs):
        depth = json.loads(cmd[-1])["depth"]
        depths.append(depth)
        if depth == 7:
            return subprocess.CompletedProcess(cmd, 0, "", "[ERROR] Failed to close loop")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(poisson.subprocess, "run", flaky_run)

    content, metadata = poisson.poisson_mesh(_sphere_cloud(), format="obj", depth=7)

    assert depths == [7, 6]
    assert metadata["depth"] == 6
    assert content.startswith(b"#")
