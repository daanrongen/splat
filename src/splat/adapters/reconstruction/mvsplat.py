"""MVSplat feed-forward reconstruction adapter — see registry/gaussian.py for
the catalog entry (MIT license, `dylanebert/mvsplat` weights).

Deliberately left as a stub. MVSplat itself is plain PyTorch (no custom CUDA
kernels needed to produce Gaussian parameters — the CUDA rasterizer in its
reference repo is only used for rendering novel views, not required here),
but it needs calibrated camera poses/intrinsics per input image rather than
raw photos, and ships only as a Hydra + PyTorch Lightning research
framework (`python -m src.main +experiment=... mode=test`) with no
importable inference function — every other license-clean, pose-free
alternative surveyed hard-depends on CUDA-only kernels for core inference
(TriplaneGaussian, LGM) or carries a non-commercial license with an
unconfirmed CUDA dependency (Splatt3R), and the most promising native
Apple Silicon option (gsplat-mlx) turned out to be pre-alpha: its only
differentiable rendering path is an explicit brute-force proof-of-concept,
not a real rasterizer capable of training on photograph-sized scenes.

Implementing this for real means vendoring MVSplat's cost-volume encoder as
a git dependency, extracting the bare nn.Module from its Lightning
wrapper, and adding a calibrated-input path to `splat convert`/`splat train`
(e.g. a COLMAP directory or a poses.json) — a substantially bigger, riskier
piece of work than fits this pass. The port and registry entry are real;
this adapter is the honest placeholder until that's taken on deliberately.
"""

from pathlib import Path

from splat.domain.gaussians import GaussianCloud
from splat.domain.value_objects import ModelLicense


class MVSplatBackend:
    name = "mvsplat"

    def __init__(self, *, weights_path: Path, device: str, license: ModelLicense) -> None:
        self._weights_path = weights_path
        self._device = device
        self.license = license

    def reconstruct(self, images: list[Path], *, device: str = "auto", **params) -> GaussianCloud:
        raise NotImplementedError(
            "MVSplatBackend.reconstruct is not yet implemented — MVSplat requires calibrated "
            "camera poses/intrinsics and a vendored encoder; see this module's docstring for "
            "the full survey of why no feed-forward model shipped real in this pass."
        )

    def required_image_count(self) -> tuple[int, int | None]:
        return (2, None)
