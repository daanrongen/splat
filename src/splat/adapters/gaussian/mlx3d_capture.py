"""Apple-local Gaussian reconstruction via mlx3d's capture pipeline.

This backend is optimization-based rather than feed-forward: mlx3d estimates
poses, trains a 3DGS scene, and exports a standard Gaussian PLY. It is the
local runnable default because it works on Apple Silicon through MLX/Metal.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from splat.adapters.formats.ply import PlyReader
from splat.domain.gaussians import GaussianCloud
from splat.domain.value_objects import ModelLicense


class MLX3DCaptureBackend:
    name = "mlx3d-capture"

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
        **params,
    ) -> GaussianCloud:
        try:
            from mlx3d.capture import CaptureConfig, run_capture
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
            summary = run_capture(str(input_dir), str(output_dir), config, log=lambda _msg: None)
            cloud = PlyReader().read(Path(summary["splat"]))
            cloud.metadata.source_model = self.name
            return cloud

    @classmethod
    def required_image_count(cls) -> tuple[int, int | None]:
        return (3, None)
