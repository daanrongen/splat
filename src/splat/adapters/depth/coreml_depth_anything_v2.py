"""Monocular *relative* depth estimation via Apple's CoreML export of
Depth Anything V2 Small (`apple/coreml-depth-anything-v2-small`,
Apache-2.0). Unlike `depth-pro`, this model predicts inverse depth
(disparity, higher value = closer) on an arbitrary per-image scale, not
metric depth in meters, and carries no focal length/field-of-view
estimate - `DepthMap.focal_length_px` stays `None`, which `splat mesh`
treats as a clean domain error (heightfield needs metric depth). Useful
anywhere a normalized depth ordering is enough - segmentation-adjacent
masking, depth-aware compositing - not for anything needing physical scale.
"""

from pathlib import Path

import coremltools as ct
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

from splat.adapters.formats.image import read_rgb
from splat.domain.image_space import DepthMap
from splat.domain.value_objects import ModelLicense
from splat.paths import model_cache_dir

_INPUT_SIZE = (518, 392)  # (W, H), fixed by Apple's conversion


class CoreMLDepthAnythingV2Backend:
    name = "depth-anything-v2-coreml"

    def __init__(self, *, hf_repo_id: str, license: ModelLicense, device: str = "auto") -> None:
        self._hf_repo_id = hf_repo_id
        self.license = license
        self._compute_unit = (
            ct.ComputeUnit.CPU_ONLY if device == "cpu" else ct.ComputeUnit.CPU_AND_GPU
        )
        self._model: ct.models.MLModel | None = None

    def _weights_dir(self) -> Path:
        target = (
            model_cache_dir() / "coreml-depth-anything-v2" / self._hf_repo_id.replace("/", "--")
        )
        # local_dir= materializes real files (reusing already-downloaded blobs via
        # hardlink, no re-fetch) instead of the default symlinked cache — the
        # CoreML compiler cannot stage a symlinked weight.bin during compilation.
        if not any(target.glob("*.mlpackage")):
            snapshot_download(
                repo_id=self._hf_repo_id,
                local_dir=target,
                allow_patterns=["DepthAnythingV2SmallF16.mlpackage/**"],
            )
        return target

    def _load(self) -> None:
        if self._model is not None:
            return
        package = next(self._weights_dir().glob("*.mlpackage"))
        self._model = ct.models.MLModel(str(package), compute_units=self._compute_unit)

    def estimate(self, image_path: Path, **params) -> DepthMap:
        self._load()
        image = read_rgb(image_path)
        original_size = (image.shape[1], image.shape[0])  # (W, H)

        resized = Image.fromarray(image).resize(_INPUT_SIZE)
        output = self._model.predict({"image": resized})
        depth_img = output["depth"].resize(original_size, Image.BILINEAR)
        depth = np.array(depth_img, dtype=np.float32)

        return DepthMap(
            depth=depth,
            focal_length_px=None,
            field_of_view_deg=None,
            units="disparity",
            metadata={"source_model": self._hf_repo_id},
        )
