"""Niantic's `.spz` (MIT, github.com/nianticlabs/spz): a gzipped header plus
attribute streams of 24-bit fixed-point positions and 8-bit everything else,
about 10x smaller than `.ply` with SH kept. Writes v3, the gzip layout every
current reader loads; reads v2 and v3. Packing constants mirror the reference
`load-spz.cc`, in its RUB frame, which is splat's canonical opengl frame.
"""

from __future__ import annotations

import gzip
import struct
from itertools import pairwise
from pathlib import Path

import numpy as np

from splat.domain.errors import SplatDomainError
from splat.domain.gaussians import (
    GaussianCloud,
    GaussianCloudMetadata,
    sh_rest_count,
    to_convention,
)

_MAGIC = 0x5053474E
_VERSION = 3
_HEADER = struct.Struct("<IIIBBBB")
_FRACTIONAL_BITS = 12
_COLOR_SCALE = 0.15
_SH1_BUCKET, _SH_REST_BUCKET = 1 << (8 - 5), 1 << (8 - 4)
_SQRT1_2 = np.float32(np.sqrt(0.5))
_MASK9 = (1 << 9) - 1


def _u8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def _pack_positions(means: np.ndarray) -> bytes:
    fixed = np.round(means * (1 << _FRACTIONAL_BITS)).astype(np.int32).reshape(-1)
    return (
        np.stack([fixed & 0xFF, (fixed >> 8) & 0xFF, (fixed >> 16) & 0xFF], axis=1)
        .astype(np.uint8)
        .tobytes()
    )


def _unpack_positions(data: bytes, n: int, fractional_bits: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    fixed = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)  # sign-extend 24 bits
    return (fixed.reshape(n, 3) / (1 << fractional_bits)).astype(np.float32)


def _pack_rotations(wxyz: np.ndarray) -> bytes:
    """Smallest-three: index of the largest |component|, then the other three as
    sign bit + 9-bit magnitude scaled by 1/sqrt(2), in xyzw order."""
    q = wxyz[:, [1, 2, 3, 0]].astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
    largest = np.abs(q).argmax(axis=1)
    negate = q[np.arange(len(q)), largest] < 0
    comp = largest.astype(np.uint32)
    for i in range(4):
        keep = largest != i
        negbit = ((q[:, i] < 0) ^ negate).astype(np.uint32)
        mag = (_MASK9 * (np.abs(q[:, i]) / _SQRT1_2) + 0.5).astype(np.uint32)
        comp = np.where(keep, (comp << 10) | (negbit << 9) | mag, comp)
    return comp.astype("<u4").tobytes()


def _unpack_rotations(data: bytes) -> np.ndarray:
    comp = np.frombuffer(data, dtype="<u4").astype(np.uint32)
    largest = comp >> 30
    q = np.zeros((len(comp), 4), dtype=np.float32)
    for i in range(3, -1, -1):
        skip = largest == i
        mag = (comp & _MASK9).astype(np.float32)
        value = _SQRT1_2 * mag / _MASK9 * np.where((comp >> 9) & 1, -1.0, 1.0)
        q[:, i] = np.where(skip, 0.0, value)
        comp = np.where(skip, comp, comp >> 10)
    q[np.arange(len(q)), largest] = np.sqrt(np.clip(1.0 - (q**2).sum(axis=1), 0.0, None))
    return q[:, [3, 0, 1, 2]]


def _quantize_sh(sh_rest: np.ndarray) -> np.ndarray:
    q = np.round(sh_rest * 128.0) + 128.0
    bucket = np.full(sh_rest.shape[1], _SH_REST_BUCKET)
    bucket[:3] = _SH1_BUCKET
    bucket = bucket[None, :, None]
    return np.clip((q + bucket // 2) // bucket * bucket, 0, 255).astype(np.uint8)


class SpzWriter:
    name = "spz"

    def write(self, cloud: GaussianCloud, path: Path) -> None:
        try:
            cloud = to_convention(cloud, "opengl")
        except ValueError as exc:
            raise SplatDomainError(f"Cannot write .spz: {exc}") from exc
        if cloud.sh_degree > 3:
            raise SplatDomainError(f"SPZ v3 stores SH degree <= 3, got {cloud.sh_degree}.")
        n = cloud.point_count
        streams = [
            _HEADER.pack(_MAGIC, _VERSION, n, cloud.sh_degree, _FRACTIONAL_BITS, 0, 0),
            _pack_positions(cloud.means),
            _u8(cloud.to_activated_opacities() * 255.0).tobytes(),
            _u8(cloud.sh_dc * (_COLOR_SCALE * 255.0) + 0.5 * 255.0).tobytes(),
            _u8((cloud.to_log_scales() + 10.0) * 16.0).tobytes(),
            _pack_rotations(cloud.rotations),
        ]
        if cloud.sh_rest is not None:
            streams.append(_quantize_sh(cloud.sh_rest).tobytes())
        path.write_bytes(gzip.compress(b"".join(streams)))

    def supports(self, cloud: GaussianCloud) -> list[str]:
        return ["positions to 1/4096 units, other attributes quantized to 8 bits"]


class SpzReader:
    name = "spz"

    def read(self, path: Path) -> GaussianCloud:
        try:
            data = gzip.decompress(path.read_bytes())
        except (OSError, EOFError) as exc:
            raise SplatDomainError(
                f"{path.name} is not a gzip .spz (v2/v3); v4 needs Niantic's reader."
            ) from exc
        magic, version, n, sh_degree, fractional_bits, _, _ = _HEADER.unpack_from(data)
        if magic != _MAGIC or version not in (2, 3):
            raise SplatDomainError(f"{path.name}: unsupported .spz (magic/version {version}).")

        k = sh_rest_count(sh_degree)
        rot_bytes = 4 if version >= 3 else 3
        sizes = [n * 9, n, n * 3, n * 3, n * rot_bytes, n * k * 3]
        offsets = np.cumsum([_HEADER.size, *sizes])
        chunks = [data[start:end] for start, end in pairwise(offsets)]
        positions, alphas, colors, scales, rotations, sh = chunks

        alpha = np.frombuffer(alphas, np.uint8).astype(np.float32) / 255.0
        if version >= 3:
            quats = _unpack_rotations(rotations)
        else:
            xyz = np.frombuffer(rotations, np.uint8).reshape(n, 3).astype(np.float32) / 127.5 - 1
            w = np.sqrt(np.clip(1.0 - (xyz**2).sum(axis=1), 0.0, None))
            quats = np.column_stack([w, xyz])
        return GaussianCloud(
            means=_unpack_positions(positions, n, fractional_bits),
            scales=(np.frombuffer(scales, np.uint8).reshape(n, 3) / 16.0 - 10.0).astype(np.float32),
            rotations=quats.astype(np.float32),
            opacities=alpha.clip(1e-6, 1 - 1e-6),
            sh_dc=(
                (np.frombuffer(colors, np.uint8).reshape(n, 3) / 255.0 - 0.5) / _COLOR_SCALE
            ).astype(np.float32),
            sh_rest=(np.frombuffer(sh, np.uint8).reshape(n, k, 3).astype(np.float32) - 128.0)
            / 128.0
            if k
            else None,
            sh_degree=sh_degree,
            scale_activation="log",
            opacity_activation="linear",
            metadata=GaussianCloudMetadata(source_format="spz"),
        )
