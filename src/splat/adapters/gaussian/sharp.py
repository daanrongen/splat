"""Apple SHARP: single-image feed-forward 3DGS reconstruction (arXiv:2512.10685).

This is the only backend that turns one image into Gaussians, so it's what
makes `splat diffuse ... | splat gaussian -` possible at all.

The published checkpoint is a bare 1038-tensor state_dict with no config, so
the architecture has to come from Apple's own model code; `_vendor/sharp/`
carries the inference subset (models + the utils they need), following the
same pattern as the mlx_stable_diffusion and mlx_sam vendors. The reference
`sharp predict` CLI is not usable here: it imports gsplat at module load for
its CUDA-only video renderer, and pulls in matplotlib/imageio/pillow-heif
that nothing on the prediction path needs.

Output is metric with absolute scale, in the OpenCV/COLMAP convention, with
the scene centered roughly at +z.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from splat.adapters.formats.image import read_rgb
from splat.adapters.formats.ply import PlyReader
from splat.domain.errors import ReconstructionBackendError
from splat.domain.gaussians import GaussianCloud
from splat.domain.value_objects import ModelLicense

# Fixed by training; the predictor regresses one Gaussian per 2x2 block of
# this grid, so a 512x512 input still yields ~1.18M Gaussians.
_INTERNAL_SIZE = 1536

# 35mm-equivalent focal length assumed when an image carries no EXIF, matching
# the reference CLI's fallback. Diffused PNGs never have it.
_DEFAULT_FOCAL_35MM = 30.0

# sigmoid(16.1) rounds to 1.0 in float32, so Apple's `save_ply` writes +inf
# opacity logits for saturated Gaussians (11,856 of 1,179,648 on a real
# prediction) and GaussianCloud rejects the file. Clamping just inside the
# open interval is lossless at float32 and keeps the export readable.
_OPACITY_EPS = 1e-7


def _resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        if device not in ("cpu", "mps", "cuda"):
            raise ReconstructionBackendError(
                f"SHARP device must be 'auto', 'cpu', 'mps' or 'cuda', got {device!r}."
            )
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _focal_px(image_path: Path, width: int, height: int, focal_35mm: float) -> float:
    """35mm-equivalent focal length in pixels for this frame's diagonal."""
    return focal_35mm * math.sqrt(width**2 + height**2) / math.sqrt(36**2 + 24**2)


class SharpBackend:
    name = "sharp"

    # SHARP predicts in metres, which is the one thing it uniquely provides;
    # run_gaussian must not recenter and rescale it away.
    provides_metric_scale = True

    def __init__(
        self, *, weights_path: Path | None = None, device: str = "auto", license: ModelLicense
    ) -> None:
        self._weights_path = weights_path
        self._device = device
        self.license = license
        self._predictor = None

    @classmethod
    def required_image_count(cls) -> tuple[int, int | None]:
        return (1, 1)

    def _checkpoint(self) -> Path:
        if self._weights_path is None:
            raise ReconstructionBackendError(
                "SHARP needs its checkpoint. Run `splat models pull sharp`."
            )
        try:
            return next(iter(sorted(self._weights_path.glob("*.pt"))))
        except StopIteration as exc:
            raise ReconstructionBackendError(
                f"No SHARP .pt checkpoint under {self._weights_path}. "
                "Run `splat models pull sharp` to fetch it."
            ) from exc

    def _load(self, device: str):
        if self._predictor is not None:
            return self._predictor

        import torch

        from splat.adapters.gaussian._vendor.sharp.models import (
            PredictorParams,
            create_predictor,
        )

        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(torch.load(self._checkpoint(), weights_only=True))
        predictor.eval().to(device)
        self._predictor = predictor
        return predictor

    def reconstruct(
        self,
        images: list[Path],
        *,
        device: str = "auto",
        focal_35mm: float = _DEFAULT_FOCAL_35MM,
        **params,
    ) -> GaussianCloud:
        if len(images) != 1:
            raise ReconstructionBackendError(
                f"SHARP reconstructs from exactly one image, got {len(images)}."
            )

        import torch

        from splat.adapters.gaussian._vendor.sharp.utils.gaussians import (
            save_ply,
            unproject_gaussians,
        )

        resolved_device = _resolve_device(device if device != "auto" else self._device)
        predictor = self._load(resolved_device)

        image = read_rgb(images[0])
        height, width = image.shape[:2]
        f_px = _focal_px(images[0], width, height, focal_35mm)

        with torch.no_grad():
            gaussians = self._predict(predictor, image, f_px, resolved_device, unproject_gaussians)
            gaussians = gaussians._replace(
                opacities=gaussians.opacities.clamp(_OPACITY_EPS, 1.0 - _OPACITY_EPS)
            )

            with tempfile.TemporaryDirectory() as tmp_dir:
                ply_path = Path(tmp_dir) / "sharp.ply"
                save_ply(gaussians, f_px, (height, width), ply_path)
                cloud = PlyReader().read(ply_path)

        cloud.metadata.source_model = self.name
        # SHARP predicts in the input camera's own frame, so the origin looking
        # down +z *is* the capture pose - unlike SfM, where camera[0] being the
        # world origin makes the same values meaningless. Recording it lets
        # `splat render` frame the cloud from the viewpoint it was seen from.
        cloud.metadata.capture_camera_position = [0.0, 0.0, 0.0]
        cloud.metadata.capture_camera_rotation = np.eye(3).tolist()
        cloud.metadata.capture_camera_intrinsics = [
            f_px,
            f_px,
            width / 2.0,
            height / 2.0,
            float(width),
            float(height),
        ]
        cloud.metadata.capture_camera_count = 1
        return cloud

    @staticmethod
    def _predict(predictor, image: np.ndarray, f_px: float, device: str, unproject_gaussians):
        """Mirrors `sharp.cli.predict.predict_image`: predict in NDC at the
        training resolution, then unproject with intrinsics scaled to it."""
        import torch
        import torch.nn.functional as functional

        height, width = image.shape[:2]
        image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
        resized = functional.interpolate(
            image_pt[None],
            size=(_INTERNAL_SIZE, _INTERNAL_SIZE),
            mode="bilinear",
            align_corners=True,
        )
        disparity_factor = torch.tensor([f_px / width]).float().to(device)
        gaussians_ndc = predictor(resized, disparity_factor)

        intrinsics = (
            torch.tensor(
                [
                    [f_px, 0, width / 2, 0],
                    [0, f_px, height / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            .float()
            .to(device)
        )
        intrinsics[0] *= _INTERNAL_SIZE / width
        intrinsics[1] *= _INTERNAL_SIZE / height
        return unproject_gaussians(
            gaussians_ndc,
            torch.eye(4).to(device),
            intrinsics,
            (_INTERNAL_SIZE, _INTERNAL_SIZE),
        )
