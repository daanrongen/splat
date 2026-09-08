"""`.sog` reader/writer: a spatially-sorted, image-codec-compressed Gaussian
splat bundle, inspired by Self-Organizing Gaussians (SOG,
fraunhoferhhi/Self-Organizing-Gaussians, ECCV'24) and PlayCanvas's real SOG
format - see #65. Two deliberate simplifications relative to both: points
are ordered by a 3D Morton (Z-order) curve rather than PLAS's differentiable
grid optimization (no per-scene training loop, no vendored code - simpler
and license-clean, at some cost to sort quality), and each attribute plane
is a plain 8/16-bit PNG rather than WebP + per-attribute codebooks (fewer
moving parts, still real compression: PNG's DEFLATE benefits substantially
from Morton-order locality on top of vanilla `.splat`-style quantization).
Higher-order SH is always dropped, like `compress`'s existing
`web-delivery` profile - this is a delivery format, not an archival one.

Bundle layout (a zip archive):
    meta.json    - count, grid width/height, sh_degree, per-plane min/max
    means_l.png  - RGB, low byte of each axis's 16-bit normalized position
    means_u.png  - RGB, high byte of each axis's 16-bit normalized position
    scales.png   - RGB, 8-bit normalized log-scale per axis
    quats.png    - RGBA, 8-bit normalized rotation quaternion (w, x, y, z)
    sh0.png      - RGBA, SH-degree-0 color (RGB) and opacity (A)
"""

from __future__ import annotations

import json
import zipfile
from math import isqrt
from pathlib import Path

import numpy as np

from splat.adapters.formats.image import decode_rgb_or_rgba, encode_png
from splat.domain.errors import SplatDomainError
from splat.domain.gaussians import GaussianCloud, GaussianCloudMetadata, to_convention

SH_C0 = 0.28209479177387814


def morton_order(points: np.ndarray, bits: int = 10) -> np.ndarray:
    """Z-order-curve permutation over quantized 3D positions - spatially
    close points end up close together in raster order, which is what lets
    the per-attribute PNG planes compress well."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    scale = (1 << bits) - 1
    q = np.clip(np.round((points - mins) / span * scale), 0, scale).astype(np.uint64)

    codes = np.zeros(points.shape[0], dtype=np.uint64)
    for axis in range(3):
        for bit in range(bits):
            codes |= ((q[:, axis] >> bit) & 1) << (bit * 3 + axis)
    return np.argsort(codes, kind="stable")


def _grid_side(n: int) -> int:
    side = isqrt(n)
    return side if side * side >= n else side + 1


def _normalize_to_uint(values: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bits: int):
    span = np.where(maxs > mins, maxs - mins, 1.0)
    scale = (1 << bits) - 1
    return np.clip(np.round((values - mins) / span * scale), 0, scale)


def _denormalize(quantized: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bits: int):
    span = maxs - mins
    scale = (1 << bits) - 1
    return quantized / scale * span + mins


class SogWriter:
    name = "sog"

    def write(self, cloud: GaussianCloud, path: Path) -> None:
        # Same fixed-convention situation as `.splat`: no field to store
        # `coordinate_convention`, so the reader always reports "opengl" -
        # actually converting keeps that true instead of just relabelling.
        try:
            cloud = to_convention(cloud, "opengl")
        except ValueError as exc:
            raise SplatDomainError(f"Cannot write .sog: {exc}") from exc
        n = cloud.point_count
        order = morton_order(cloud.means)
        side = _grid_side(n)
        cells = side * side

        means = cloud.means[order]
        log_scales = cloud.to_log_scales()[order]
        rotations = cloud.rotations[order]
        colors = np.clip(0.5 + SH_C0 * cloud.sh_dc[order], 0.0, 1.0)
        opacities = np.clip(cloud.to_activated_opacities()[order], 0.0, 1.0)

        means_mins, means_maxs = means.min(axis=0), means.max(axis=0)
        scale_mins, scale_maxs = log_scales.min(axis=0), log_scales.max(axis=0)

        means_q = _normalize_to_uint(means, means_mins, means_maxs, 16).astype(np.uint16)
        means_l = np.zeros((cells, 3), dtype=np.uint8)
        means_u = np.zeros((cells, 3), dtype=np.uint8)
        means_l[:n] = (means_q & 0xFF).astype(np.uint8)
        means_u[:n] = (means_q >> 8).astype(np.uint8)

        scales_q = np.zeros((cells, 3), dtype=np.uint8)
        scales_q[:n] = _normalize_to_uint(log_scales, scale_mins, scale_maxs, 8).astype(np.uint8)

        norm = np.linalg.norm(rotations, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        quats_q = np.zeros((cells, 4), dtype=np.uint8)
        quats_q[:n] = np.clip(np.round((rotations / norm) * 128.0 + 128.0), 0, 255).astype(np.uint8)

        sh0_q = np.zeros((cells, 4), dtype=np.uint8)
        sh0_q[:n, :3] = np.round(colors * 255.0).astype(np.uint8)
        sh0_q[:n, 3] = np.round(opacities * 255.0).astype(np.uint8)

        meta = {
            "version": 1,
            "count": n,
            "width": side,
            "height": side,
            "means": {"mins": means_mins.tolist(), "maxs": means_maxs.tolist()},
            "scales": {"mins": scale_mins.tolist(), "maxs": scale_maxs.tolist()},
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json", json.dumps(meta))
            zf.writestr("means_l.png", encode_png(means_l.reshape(side, side, 3)))
            zf.writestr("means_u.png", encode_png(means_u.reshape(side, side, 3)))
            zf.writestr("scales.png", encode_png(scales_q.reshape(side, side, 3)))
            zf.writestr("quats.png", encode_png(quats_q.reshape(side, side, 4)))
            zf.writestr("sh0.png", encode_png(sh0_q.reshape(side, side, 4)))

    def supports(self, cloud: GaussianCloud) -> list[str]:
        warnings = [
            "positions quantized to 16 bits/axis, scale/rotation/color/opacity to 8 bits/channel"
        ]
        if cloud.sh_degree > 0:
            warnings.append(
                f"SH degree {cloud.sh_degree} -> 0 (higher-order spherical harmonics dropped)"
            )
        return warnings


class SogReader:
    name = "sog"

    def read(self, path: Path) -> GaussianCloud:
        with zipfile.ZipFile(path) as zf:
            meta = json.loads(zf.read("meta.json"))
            means_l = decode_rgb_or_rgba(zf.read("means_l.png"))
            means_u = decode_rgb_or_rgba(zf.read("means_u.png"))
            scales_img = decode_rgb_or_rgba(zf.read("scales.png"))
            quats_img = decode_rgb_or_rgba(zf.read("quats.png"))
            sh0_img = decode_rgb_or_rgba(zf.read("sh0.png"))

        n = meta["count"]

        means_q = (
            means_l.reshape(-1, 3)[:n].astype(np.uint16)
            | means_u.reshape(-1, 3)[:n].astype(np.uint16) << 8
        )
        means_mins = np.array(meta["means"]["mins"], dtype=np.float32)
        means_maxs = np.array(meta["means"]["maxs"], dtype=np.float32)
        means = _denormalize(means_q.astype(np.float32), means_mins, means_maxs, 16).astype(
            np.float32
        )

        scale_mins = np.array(meta["scales"]["mins"], dtype=np.float32)
        scale_maxs = np.array(meta["scales"]["maxs"], dtype=np.float32)
        scales = _denormalize(
            scales_img.reshape(-1, 3)[:n].astype(np.float32), scale_mins, scale_maxs, 8
        ).astype(np.float32)

        rotations = ((quats_img.reshape(-1, 4)[:n].astype(np.float32) - 128.0) / 128.0).astype(
            np.float32
        )

        sh0 = sh0_img.reshape(-1, 4)[:n].astype(np.float32) / 255.0
        sh_dc = ((sh0[:, :3] - 0.5) / SH_C0).astype(np.float32)
        opacities = sh0[:, 3].astype(np.float32)

        return GaussianCloud(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_dc=sh_dc,
            sh_rest=None,
            sh_degree=0,
            scale_activation="log",
            opacity_activation="linear",
            metadata=GaussianCloudMetadata(source_format="sog"),
        )
