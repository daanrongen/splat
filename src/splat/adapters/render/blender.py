"""Blender-as-render-engine adapter: shells out to `blender --background
--python _blender_script.py` to render a GaussianCloud as a native Blender
point cloud (see that script for the actual scene setup).

v2 renders true oriented/scaled ellipsoids (rotation quaternion + per-axis
scale) alpha-blended by opacity, on Cycles by default - EEVEE remains
available as a faster, approximate preview (`--engine eevee`). Defaults to
a real captured camera pose (see #59) when the cloud has one, falling back
to v1's auto-fit heuristic otherwise. See issue #50 for the v1 scope cut
this replaces, and #58 for the motivating research.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from splat.domain.errors import RenderBackendError
from splat.domain.gaussians import GaussianCloud
from splat.domain.value_objects import convention_flip_matrix

_SCRIPT_PATH = Path(__file__).parent / "_blender_script.py"
_SH_C0 = 0.28209479177387814
_DEFAULT_TIMEOUT_SECONDS = 1800.0

# COLMAP/OpenCV world convention (+X right, +Y down, +Z forward) -> Blender/OpenGL
# (+X right, +Y up, -Z forward). diag(1,-1,-1) has determinant +1: this is a proper
# 180-degree rotation about X, not a mirroring, so it applies cleanly to positions,
# per-Gaussian orientations, and camera poses alike.
_COLMAP_TO_BLENDER_FLIP = convention_flip_matrix("colmap", "opengl")


def _resolve_blender_bin(explicit: str | None) -> str:
    candidate = explicit or os.environ.get("SPLAT_BLENDER_BIN") or shutil.which("blender")
    if candidate is None:
        raise RenderBackendError(
            "Could not find a `blender` executable. Install Blender and ensure it's on PATH, "
            "or set SPLAT_BLENDER_BIN / pass blender_bin=<path>."
        )
    return candidate


def _resolve_timeout() -> float | None:
    """0 or a negative value disables the timeout, for very large clouds."""
    seconds = float(os.environ.get("SPLAT_RENDER_TIMEOUT", _DEFAULT_TIMEOUT_SECONDS))
    return seconds if seconds > 0 else None


def _convert_positions(points: np.ndarray) -> np.ndarray:
    return (points @ _COLMAP_TO_BLENDER_FLIP).astype(np.float32)


def _convert_gaussian_rotations(quats: np.ndarray) -> np.ndarray:
    """Conjugating a (w,x,y,z) quaternion by the 180-about-X flip works out to
    negating just the y,z components - derived from the Hamilton product, not
    guessed (a matrix round-trip gives the identical result, checked in tests)."""
    return (quats * np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)).astype(np.float32)


def _convert_camera_rotation(rotation_colmap: list[list[float]]) -> np.ndarray:
    """World-to-camera -> camera-to-world, then conjugate by the flip on both the
    world side (left) and the camera's own local-axis convention side (right) -
    COLMAP cameras look down local +Z with +Y down; Blender cameras look down
    local -Z with +Y up, also related by the same 180-about-X rotation."""
    r_world_to_cam = np.asarray(rotation_colmap, dtype=np.float32)
    r_cam_to_world = r_world_to_cam.T
    return (_COLMAP_TO_BLENDER_FLIP @ r_cam_to_world @ _COLMAP_TO_BLENDER_FLIP).astype(np.float32)


def _cloud_to_npz(cloud: GaussianCloud, path: Path) -> None:
    colmap = cloud.metadata.coordinate_convention == "colmap"
    means = _convert_positions(cloud.means) if colmap else cloud.means.astype(np.float32)
    rotations = (
        _convert_gaussian_rotations(cloud.rotations)
        if colmap
        else cloud.rotations.astype(np.float32)
    )
    colors = np.clip(0.5 + _SH_C0 * cloud.sh_dc, 0.0, 1.0).astype(np.float32)
    scales = cloud.to_linear_scales().astype(np.float32)
    opacities = cloud.to_activated_opacities().astype(np.float32)

    arrays = {
        "means": means,
        "colors": colors,
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities,
    }

    metadata = cloud.metadata
    if (
        metadata.capture_camera_position is not None
        and metadata.capture_camera_rotation is not None
    ):
        position = np.asarray(metadata.capture_camera_position, dtype=np.float32)
        arrays["camera_position"] = _convert_positions(position) if colmap else position
        rotation = metadata.capture_camera_rotation
        arrays["camera_rotation"] = (
            _convert_camera_rotation(rotation) if colmap else np.asarray(rotation, dtype=np.float32)
        )
        if metadata.capture_camera_intrinsics is not None:
            arrays["camera_intrinsics"] = np.asarray(
                metadata.capture_camera_intrinsics, dtype=np.float32
            )

    np.savez(path, **arrays)


class BlenderBackend:
    name = "blender"

    def render(
        self,
        cloud: GaussianCloud,
        output_path: Path,
        *,
        width: int = 1280,
        height: int = 720,
        samples: int = 32,
        engine: str = "cycles",
        blender_bin: str | None = None,
        **params,
    ) -> None:
        blender = _resolve_blender_bin(blender_bin)

        with tempfile.TemporaryDirectory() as tmp_dir:
            npz_path = Path(tmp_dir) / "cloud.npz"
            _cloud_to_npz(cloud, npz_path)

            command = [
                blender,
                "--background",
                "--factory-startup",
                "--python",
                str(_SCRIPT_PATH),
                "--",
                "--input",
                str(npz_path),
                "--output",
                str(output_path),
                "--width",
                str(width),
                "--height",
                str(height),
                "--samples",
                str(samples),
                "--engine",
                engine,
            ]
            timeout = _resolve_timeout()
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise RenderBackendError(
                    f"Blender render exceeded {timeout:.0f}s. Lower --samples, use "
                    "--engine eevee, or raise SPLAT_RENDER_TIMEOUT (0 disables it)."
                ) from exc

        if result.returncode != 0 or not output_path.exists():
            raise RenderBackendError(
                f"Blender render failed (exit {result.returncode}):\n{result.stderr}"
            )
