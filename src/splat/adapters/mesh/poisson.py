"""Poisson-surface-reconstruction mesh extraction (SuGaR/GOF-inspired scope,
without either paper's training loop or INRIA-lineage code - see #69).
Screened Poisson reconstruction (Kazhdan & Hoppe) is a generic classical
algorithm, license-clean independent of any reference repo; Open3D (MIT)
provides a maintained CPU implementation.
"""

import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import open3d as o3d

from splat.adapters.formats.splat_fmt import SH_C0
from splat.domain.errors import SplatDomainError, UnsupportedFormat
from splat.domain.gaussians import GaussianCloud

SUPPORTED_FORMATS = ("obj", "glb", "gltf")
MIN_POINTS = 4


class PoissonMeshExporter:
    name = "poisson"

    def export(
        self,
        cloud: GaussianCloud,
        path: Path,
        *,
        format: str,
        depth: int = 9,
        opacity_threshold: float = 0.1,
        density_quantile: float = 0.05,
        **params,
    ) -> tuple[int, int]:
        fmt = format.lstrip(".").lower()
        if fmt not in SUPPORTED_FORMATS:
            raise UnsupportedFormat(f"poisson meshing supports {SUPPORTED_FORMATS}, got {fmt!r}")

        keep = cloud.to_activated_opacities() >= opacity_threshold
        if keep.sum() < MIN_POINTS:
            keep = np.ones(cloud.point_count, dtype=bool)
        means = cloud.means[keep]
        colors = np.clip(0.5 + SH_C0 * cloud.sh_dc[keep], 0.0, 1.0)
        if means.shape[0] < MIN_POINTS:
            raise SplatDomainError(
                f"poisson meshing needs at least {MIN_POINTS} points, got {means.shape[0]}"
            )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(means.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=min(16, means.shape[0] - 1))

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )
        threshold = np.quantile(np.asarray(densities), density_quantile)
        mesh.remove_vertices_by_mask(np.asarray(densities) < threshold)
        mesh.compute_vertex_normals()

        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(path), mesh)

        return len(mesh.vertices), len(mesh.triangles)


def _export_in_child(
    cloud_path: Path, out_path: Path, format: str, params: dict
) -> tuple[int, int]:
    from splat.adapters.formats.ply import PlyReader

    return PoissonMeshExporter().export(
        PlyReader().read(cloud_path), out_path, format=format, **params
    )


def poisson_mesh(cloud: GaussianCloud, *, format: str, **params) -> tuple[bytes, int, int]:
    """Runs Poisson in a fresh process: Open3D's OpenMP segfaults when it shares
    one with torch's (#85)."""
    from splat.adapters.formats.ply import PlyWriter

    with tempfile.TemporaryDirectory() as tmp_dir:
        cloud_path, out_path = Path(tmp_dir) / "cloud.ply", Path(tmp_dir) / f"mesh.{format}"
        PlyWriter().write(cloud, cloud_path)
        spawn = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=1, mp_context=spawn) as pool:
                job = pool.submit(_export_in_child, cloud_path, out_path, format, params)
                vertices, faces = job.result()
        except BrokenProcessPool as exc:
            raise SplatDomainError("Poisson meshing crashed in its worker process.") from exc
        return out_path.read_bytes(), vertices, faces
