"""TripoSR feed-forward image-to-mesh adapter - see registry/mesh.py for the
catalog entry (MIT license, `stabilityai/TripoSR` weights).

Deliberately left as a stub. No Apple-native or depth-conditioned
single-image-to-mesh model exists anywhere surveyed (Apple's HF org ships
nothing image-to-3D; `microsoft/TRELLIS.2-4B` is the closest recent option
but hard-requires CUDA 12.4 + a 24GB-VRAM NVIDIA GPU on Linux). TripoSR is
the most viable real candidate: plain `torch` forward pass (MPS-compatible),
a genuine importable `from_pretrained`-style API, and a marching-cubes
dependency (`torchmcubes`) with a CPU-only build path.

It is image-only, though - it does not accept a depth map. Nothing surveyed
conditions mesh prediction on depth; the only place a depth map could
honestly participate is post-hoc rescaling of TripoSR's arbitrary-scale
output into metric units via a known focal length, which is a separate
concern from prediction itself and not implemented here.

Implementing this for real means vendoring TripoSR's `tsr` package, which is
not published on PyPI, and wiring `torchmcubes`' CPU fallback into packaging.
"""

from pathlib import Path

from splat.domain.image_space import Shape3D
from splat.domain.value_objects import ModelLicense


class TripoSRBackend:
    name = "triposr"

    def __init__(self, *, hf_repo_id: str, license: ModelLicense, device: str = "auto") -> None:
        self._hf_repo_id = hf_repo_id
        self.license = license
        self._device = device

    def predict(self, image_path: Path, **params) -> Shape3D:
        raise NotImplementedError(
            "TripoSRBackend.predict is not yet implemented - TripoSR's `tsr` package isn't "
            "vendored yet and its torchmcubes dependency needs a CPU-only build wired into "
            "packaging."
        )
