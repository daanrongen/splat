"""TripoSR feed-forward image-to-mesh adapter — see registry/mesh.py for the
catalog entry (MIT license, `stabilityai/TripoSR` weights).

Deliberately left as a stub. No Apple-native or depth-conditioned
single-image-to-mesh model exists anywhere surveyed (Apple's HF org ships
nothing image-to-3D; `microsoft/TRELLIS.2-4B` is the closest recent option
but hard-requires CUDA 12.4 + a 24GB-VRAM NVIDIA GPU on Linux, the same
class of disqualifier that ruled out gsplat-mlx/TriplaneGaussian for
`gaussian`). TripoSR is the most viable real candidate: plain `torch`
forward pass (MPS-compatible), a genuine importable `from_pretrained`-style
API rather than a Hydra/Lightning research harness, and its marching-cubes
dependency (`torchmcubes`) builds CPU-only when no CUDA compiler is present
per its own CMakeLists.txt, despite what its README implies.

It is image-only, though — it does not accept a depth map. Nothing surveyed
conditions mesh prediction on depth; the only place a depth map could
honestly participate is post-hoc rescaling of TripoSR's arbitrary-scale
output into metric units via a known focal length, which is a separate
concern from prediction itself and not implemented here.

Implementing this for real means vendoring TripoSR's `tsr` package (not on
PyPI) and wiring `torchmcubes`' CPU fallback build into this project's
packaging — real work, but nowhere near MVSplat's scope.
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
            "TripoSRBackend.predict is not yet implemented — TripoSR's `tsr` package isn't "
            "vendored yet and its torchmcubes dependency needs a CPU-only build wired into "
            "packaging; see this module's docstring for the full survey of why no image-to-mesh "
            "model shipped real in this pass."
        )
