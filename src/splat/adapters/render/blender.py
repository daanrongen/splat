"""Blender-as-render-engine adapter: shells out to `blender --background
--python _blender_script.py` to render a GaussianCloud as a native Blender
point cloud (see that script for the actual scene setup).

Gaussians are rendered the way 3DGS defines them: unlit emission, alpha
composited, with each kernel's `exp(-0.5 * m^2)` falloff evaluated per ray
from the inverse-covariance basis this module derives (`_gaussian_axes`).
Cycles by default; EEVEE remains available as a faster, approximate preview
(`--engine eevee`). Defaults to a real captured camera pose (see #59) when
the cloud has one, falling back to an auto-fit heuristic otherwise. See
issue #82 for the lit-opaque-spheres model this replaces, #50 for the v1
scope cut, and #58 for the motivating research.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from splat.domain.errors import RenderBackendError
from splat.domain.gaussians import GaussianCloud
from splat.domain.value_objects import convention_flip_matrix

_SCRIPT_PATH = Path(__file__).parent / "_blender_script.py"
_SH_C0 = 0.28209479177387814
_DEFAULT_TIMEOUT_SECONDS = 1800.0
_REPORT_PREFIX = "splat| "
# Kernel support: a Gaussian is cut off at 3 sigma, where its own falloff has
# already dropped it to exp(-4.5) ~= 1% of centre opacity.
_SIGMA_SUPPORT = 3.0
_MIN_SIGMA = 1e-8

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


def _srgb_to_linear(colors: np.ndarray) -> np.ndarray:
    """3DGS SH coefficients are fit against sRGB-encoded training images, so the
    activated DC term is a display value, not radiance. Blender's shaders and
    the Standard view transform both work in linear light, so handing the
    display value straight to an Emission node re-encodes it twice and washes
    the whole frame out."""
    return np.where(colors <= 0.04045, colors / 12.92, ((colors + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )


def _gaussian_axes(quats: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """(N, 3, 3): the rows of `S^-1 R^T` per Gaussian, i.e. the map from a world
    offset to the space where that Gaussian is the unit isotropic sphere. The
    render shader needs the inverse covariance, not the covariance, and this is
    its square root - the form that stays a plain dot product per axis."""
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    columns = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1),
            np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=1),
            np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1),
        ],
        axis=1,
    )
    return (columns / np.maximum(scales, _MIN_SIGMA)[:, :, None]).astype(np.float32)


def _convert_camera_rotation(
    rotation_world_to_camera: list[list[float]], *, colmap: bool = True
) -> np.ndarray:
    """World-to-camera -> the camera-to-world matrix Blender wants on the object.

    The transpose applies either way; only the axis flip is conditional. For a
    COLMAP cloud it conjugates on both sides: the world side (left) and the
    camera's own local-axis convention (right), since COLMAP cameras look down
    local +Z with +Y down while Blender's look down local -Z with +Y up - the
    same 180-degree rotation about X.
    """
    r_cam_to_world = np.asarray(rotation_world_to_camera, dtype=np.float32).T
    if not colmap:
        return r_cam_to_world.astype(np.float32)
    return (_COLMAP_TO_BLENDER_FLIP @ r_cam_to_world @ _COLMAP_TO_BLENDER_FLIP).astype(np.float32)


def _parse_look_at(spec: str | None) -> str | None:
    """Normalizes `--look-at` and rejects it here rather than inside Blender,
    where a bad value surfaces as a subprocess exit code and a log tail."""
    if spec is None:
        return None
    try:
        values = [float(part) for part in spec.replace(" ", "").split(",")]
    except ValueError as exc:
        raise RenderBackendError(
            f"--look-at expects three comma-separated numbers, got {spec!r}"
        ) from exc
    if len(values) != 3:
        raise RenderBackendError(f"--look-at expects three coordinates, got {len(values)}")
    return ",".join(str(value) for value in values)


def _cloud_to_npz(cloud: GaussianCloud, path: Path) -> None:
    colmap = cloud.metadata.coordinate_convention == "colmap"
    means = _convert_positions(cloud.means) if colmap else cloud.means.astype(np.float32)
    rotations = (
        _convert_gaussian_rotations(cloud.rotations)
        if colmap
        else cloud.rotations.astype(np.float32)
    )
    colors = _srgb_to_linear(np.clip(0.5 + _SH_C0 * cloud.sh_dc, 0.0, 1.0))
    scales = cloud.to_linear_scales().astype(np.float32)
    opacities = cloud.to_activated_opacities().astype(np.float32)

    arrays = {
        "means": means,
        "colors": colors,
        "opacities": opacities,
        "axes": _gaussian_axes(rotations, scales),
        # One uniform billboard radius per Gaussian: the shader evaluates the
        # anisotropic falloff, so the proxy quad only has to be large enough to
        # cover the widest axis.
        "radii": (_SIGMA_SUPPORT * scales.max(axis=1)).astype(np.float32),
    }

    metadata = cloud.metadata
    if (
        metadata.capture_camera_position is not None
        and metadata.capture_camera_rotation is not None
    ):
        position = np.asarray(metadata.capture_camera_position, dtype=np.float32)
        arrays["camera_position"] = _convert_positions(position) if colmap else position
        arrays["camera_rotation"] = _convert_camera_rotation(
            metadata.capture_camera_rotation, colmap=colmap
        )
        if metadata.capture_camera_intrinsics is not None:
            arrays["camera_intrinsics"] = np.asarray(
                metadata.capture_camera_intrinsics, dtype=np.float32
            )

    np.savez(path, **arrays)


def _stream(
    command: list[str], timeout: float | None, on_progress: Callable[[str], None] | None
) -> tuple[int, list[str]]:
    """Run Blender, forwarding its progress instead of swallowing it.

    A 1.18M-Gaussian frame at 1920x1080/64 takes about 100 seconds and Blender
    prints nothing at all in background mode, so the old `capture_output=True`
    made every slow render look like a hang. `_blender_script.py` reports
    through a `render_stats` handler; those lines are forwarded and the rest is
    kept only as a tail for the failure message.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    tail: deque[str] = deque(maxlen=40)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip()
        tail.append(line)
        if on_progress is not None and line.startswith(_REPORT_PREFIX):
            on_progress(line[len(_REPORT_PREFIX) :])
        if deadline is not None and time.monotonic() > deadline:
            process.kill()
            process.wait()
            raise RenderBackendError(_timeout_message(timeout))
    try:
        remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        return process.wait(timeout=remaining), list(tail)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        raise RenderBackendError(_timeout_message(timeout)) from exc


def _timeout_message(timeout: float | None) -> str:
    return (
        f"Blender render exceeded {timeout:.0f}s. Lower --samples, use --engine eevee, "
        "thin the cloud with `splat export --profile web-delivery`, or raise "
        "SPLAT_RENDER_TIMEOUT (0 disables it)."
    )


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
        background: str = "black",
        azimuth: float | None = None,
        elevation: float | None = None,
        distance: float | None = None,
        zoom: float | None = None,
        fov: float | None = None,
        look_at: str | None = None,
        blender_bin: str | None = None,
        on_progress: Callable[[str], None] | None = None,
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
                "--background-color",
                background,
            ]
            # Only what the caller actually asked for: the script needs to
            # distinguish "no viewpoint given" (use the capture pose) from an
            # explicit one, and a defaulted value would erase that.
            for name, value in (
                ("azimuth", azimuth),
                ("elevation", elevation),
                ("distance", distance),
                ("zoom", zoom),
                ("fov", fov),
                ("look-at", _parse_look_at(look_at)),
            ):
                if value is not None:
                    command += [f"--{name}", str(value)]
            returncode, tail = _stream(command, _resolve_timeout(), on_progress)

        if returncode != 0 or not output_path.exists():
            raise RenderBackendError(
                f"Blender render failed (exit {returncode}):\n" + "\n".join(tail)
            )
