"""Blender-as-render-engine adapter: shells out to `blender --background
--python _blender_script.py` to render a GaussianCloud as a native Blender
point cloud (see that script for the actual scene setup).

v1 renders flat-colored point splats (position + SH DC color + a uniform
per-point radius derived from scale) - no ellipsoid orientation from the
rotation quaternion, no opacity blending, no view-dependent SH. That's a
deliberate scope cut (see issue #50): enough to inspect a reconstruction's
shape/orientation at a glance, not a differentiable-rasterizer replacement.
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

_SCRIPT_PATH = Path(__file__).parent / "_blender_script.py"
_SH_C0 = 0.28209479177387814


def _resolve_blender_bin(explicit: str | None) -> str:
    candidate = explicit or os.environ.get("SPLAT_BLENDER_BIN") or shutil.which("blender")
    if candidate is None:
        raise RenderBackendError(
            "Could not find a `blender` executable. Install Blender and ensure it's on PATH, "
            "or set SPLAT_BLENDER_BIN / pass blender_bin=<path>."
        )
    return candidate


def _cloud_to_npz(cloud: GaussianCloud, path: Path) -> None:
    colors = np.clip(0.5 + _SH_C0 * cloud.sh_dc, 0.0, 1.0).astype(np.float32)
    np.savez(path, means=cloud.means.astype(np.float32), colors=colors)


class BlenderRenderBackend:
    name = "blender"

    def render(
        self,
        cloud: GaussianCloud,
        output_path: Path,
        *,
        width: int = 1280,
        height: int = 720,
        samples: int = 32,
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
            ]
            result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0 or not output_path.exists():
            raise RenderBackendError(
                f"Blender render failed (exit {result.returncode}):\n{result.stderr}"
            )
