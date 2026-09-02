"""Reader/writer for antimatter15's `.splat` web-viewer format: a flat binary
array of 32-byte records, one per Gaussian:

    position   3 x float32   (world-space xyz)
    scale      3 x float32   (linear, i.e. already exp'd)
    color      4 x uint8     (RGB baked from SH degree-0 term + alpha opacity)
    rotation   4 x uint8     (quaternion w,x,y,z, packed as byte*128+128)

This format has no per-point SH-rest storage and 8-bit color/rotation
precision, so writing it is always lossy for any cloud with sh_degree > 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from splat.domain.gaussians import GaussianCloud, GaussianCloudMetadata

SH_C0 = 0.28209479177387814

_RECORD_DTYPE = np.dtype(
    [
        ("position", "<f4", 3),
        ("scale", "<f4", 3),
        ("color", "u1", 4),
        ("rotation", "u1", 4),
    ]
)


class SplatFormatReader:
    name = "splat"

    def read(self, path: Path) -> GaussianCloud:
        data = np.frombuffer(path.read_bytes(), dtype=_RECORD_DTYPE)

        means = data["position"].astype(np.float32).copy()
        scales = data["scale"].astype(np.float32).copy()

        rgba = data["color"].astype(np.float32) / 255.0
        sh_dc = ((rgba[:, :3] - 0.5) / SH_C0).astype(np.float32)
        opacities = rgba[:, 3].astype(np.float32).copy()

        rotations = ((data["rotation"].astype(np.float32) - 128.0) / 128.0).astype(np.float32)

        return GaussianCloud(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_dc=sh_dc,
            sh_rest=None,
            sh_degree=0,
            scale_activation="linear",
            opacity_activation="linear",
            metadata=GaussianCloudMetadata(source_format="splat"),
        )


class SplatFormatWriter:
    name = "splat"

    def write(self, cloud: GaussianCloud, path: Path) -> None:
        n = cloud.point_count

        colors = np.clip(0.5 + SH_C0 * cloud.sh_dc, 0.0, 1.0)
        alphas = np.clip(cloud.to_activated_opacities(), 0.0, 1.0)
        rgba_u8 = np.round(np.concatenate([colors, alphas[:, None]], axis=1) * 255.0).astype(
            np.uint8
        )

        rot = cloud.rotations.astype(np.float32)
        norm = np.linalg.norm(rot, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        rot_u8 = np.clip(np.round((rot / norm) * 128.0 + 128.0), 0, 255).astype(np.uint8)

        records = np.zeros(n, dtype=_RECORD_DTYPE)
        records["position"] = cloud.means.astype(np.float32)
        records["scale"] = cloud.to_linear_scales().astype(np.float32)
        records["color"] = rgba_u8
        records["rotation"] = rot_u8

        path.write_bytes(records.tobytes())

    def supports(self, cloud: GaussianCloud) -> list[str]:
        warnings = ["color, opacity and rotation quantized to 8 bits per channel"]
        if cloud.sh_degree > 0:
            warnings.append(
                f"SH degree {cloud.sh_degree} -> 0 (higher-order spherical harmonics dropped)"
            )
        return warnings
