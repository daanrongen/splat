"""Depth-map heightfield triangulation: back-projects a metric depth map into
a textured triangle mesh (two triangles per pixel quad), dropping faces that
straddle a large depth discontinuity so silhouette edges don't "shrink-wrap"
into the background.

No Apple-native (or any) single-image-to-mesh model exists to survey here —
see registry/mesh.py for that gap — so this leans on Depth Pro's real metric
depth instead: pure geometry, no model weights, no GPU, milliseconds on CPU.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from splat.domain.errors import SplatDomainError
from splat.domain.image_space import DepthMap, Shape3D
from splat.domain.value_objects import ModelLicense


class DepthHeightfieldBackend:
    name = "depth-heightfield"

    def __init__(self, *, license: ModelLicense, device: str = "auto") -> None:
        self.license = license

    def predict(
        self,
        image_path: Path,
        *,
        depth_map: DepthMap | None = None,
        max_depth_jump: float = 0.05,
        **params,
    ) -> Shape3D:
        if depth_map is None:
            raise SplatDomainError(
                "depth-heightfield requires a depth map — pipe through `splat depth` first, "
                "e.g. `splat generate ... | splat depth - | splat mesh -`."
            )
        if depth_map.focal_length_px is None:
            raise SplatDomainError(
                "depth-heightfield requires a depth map with a known focal length."
            )

        depth = depth_map.depth
        h, w = depth.shape
        image = np.array(Image.open(image_path).convert("RGB").resize((w, h)))

        # Approximate the principal point as the image center — DepthMap
        # doesn't carry cx/cy, only focal length and field of view.
        focal = depth_map.focal_length_px
        cx, cy = w / 2.0, h / 2.0
        ys, xs = np.mgrid[0:h, 0:w]
        x = (xs - cx) * depth / focal
        y = -(ys - cy) * depth / focal  # image rows increase downward, +y is up
        vertices = np.stack([x, y, -depth], axis=-1).reshape(-1, 3).astype(np.float32)
        uv = (
            np.stack([xs / max(w - 1, 1), 1.0 - ys / max(h - 1, 1)], axis=-1)
            .reshape(-1, 2)
            .astype(np.float32)
        )

        idx = np.arange(h * w).reshape(h, w)
        tl, tr, bl, br = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        faces = np.concatenate(
            [
                np.stack([tl, bl, tr], axis=-1).reshape(-1, 3),
                np.stack([tr, bl, br], axis=-1).reshape(-1, 3),
            ]
        )

        face_depth = depth.reshape(-1)[faces]
        jump = face_depth.max(axis=1) - face_depth.min(axis=1)
        keep = jump < max_depth_jump * face_depth.mean(axis=1)
        faces = faces[keep].astype(np.int64)

        return Shape3D(
            vertices=vertices,
            faces=faces,
            uv=uv,
            texture=image.astype(np.uint8),
            metadata={"vertex_count": len(vertices), "face_count": len(faces)},
        )
