"""Apple-local Gaussian reconstruction via mlx3d's capture pipeline.

This backend is optimization-based rather than feed-forward: mlx3d estimates
poses, trains a 3DGS scene, and exports a standard Gaussian PLY. It runs
entirely on Apple Silicon through MLX/Metal.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from splat.adapters.formats.ply import PlyReader
from splat.domain.errors import SplatDomainError
from splat.domain.gaussians import GaussianCloud
from splat.domain.value_objects import ModelLicense


class MLX3DCaptureBackend:
    name = "mlx3d-capture"

    # SfM recovers structure only up to an unknown similarity transform.
    provides_metric_scale = False

    def __init__(
        self, *, weights_path: Path | None = None, device: str = "auto", license: ModelLicense
    ) -> None:
        self._weights_path = weights_path
        self._device = device
        self.license = license

    def reconstruct(
        self,
        images: list[Path],
        *,
        device: str = "auto",
        quality: str = "fast",
        iters: int | None = None,
        max_dim: int | None = None,
        sh_degree: int | None = None,
        poses: str = "auto",
        refine_poses: str = "auto",
        low_memory: bool = False,
        seed: int = 0,
        on_progress: Callable[[str], None] | None = None,
        **params,
    ) -> GaussianCloud:
        try:
            from mlx3d.capture import CaptureConfig, run_capture
            from mlx3d.datasets.colmap import load_colmap
        except ImportError as exc:
            raise NotImplementedError(
                "The mlx3d-capture backend requires the optional mlx3d capture runtime. "
                'Install it with `uv add "mlx3d[capture]"`.'
            ) from exc

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = tmp_path / "images"
            input_dir.mkdir()
            for i, image in enumerate(images):
                suffix = image.suffix or ".png"
                (input_dir / f"{i:03d}{suffix}").write_bytes(image.read_bytes())

            output_dir = tmp_path / "capture"
            config = CaptureConfig(
                quality=quality,
                poses=poses,
                refine_poses=refine_poses,
                iters=iters,
                train_max_dim=max_dim,
                sh_degree=sh_degree,
                viewer=False,
                viewer_open_browser=False,
                low_memory=low_memory,
                seed=seed,
                overwrite=True,
            )
            try:
                summary = run_capture(
                    str(input_dir), str(output_dir), config, log=on_progress or (lambda _msg: None)
                )
            except RuntimeError as exc:
                raise SplatDomainError(f"mlx3d capture failed: {exc}") from exc
            cloud = PlyReader().read(Path(summary["splat"]))
            cloud.metadata.source_model = self.name
            # SfM triangulates in the COLMAP frame; the PLY carries no comment
            # saying so, and PlyReader's default would claim opengl.
            cloud.metadata.coordinate_convention = "colmap"
            self._attach_camera_pose(cloud, summary, output_dir, load_colmap)
            return cloud

    @classmethod
    def required_image_count(cls) -> tuple[int, int | None]:
        return (3, None)

    @staticmethod
    def _attach_camera_pose(
        cloud: GaussianCloud, summary: dict, output_dir: Path, load_colmap
    ) -> None:
        """Reads the SfM sparse model `run_capture` already wrote and stashes one real
        camera pose on the cloud's metadata (in the same COLMAP world frame as `means` -
        composes with zero conversion) instead of letting it disappear with the temp dir.
        Prefers the pose-refined sparse model when refinement ran, since that's the one
        consistent with the trained Gaussians.
        """
        refined_sparse = summary.get("train", {}).get("refined_sparse")
        colmap_root = Path(refined_sparse).parent.parent if refined_sparse else output_dir
        try:
            colmap = load_colmap(str(colmap_root), load_images=False)
        except (FileNotFoundError, OSError):
            return
        if not colmap.cameras:
            return
        cam = colmap.cameras[0]
        cloud.metadata.capture_camera_position = cam.camera_center.tolist()
        cloud.metadata.capture_camera_rotation = cam.R.tolist()
        cloud.metadata.capture_camera_intrinsics = [
            float(cam.fx),
            float(cam.fy),
            float(cam.cx),
            float(cam.cy),
            float(cam.width),
            float(cam.height),
        ]
        cloud.metadata.capture_camera_count = len(colmap.cameras)
