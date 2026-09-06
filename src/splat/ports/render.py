from pathlib import Path
from typing import Protocol

from splat.domain.gaussians import GaussianCloud


class RenderBackend(Protocol):
    """Renders a GaussianCloud to a still image using some render engine.

    An outlier among the backend ports: every sibling (ReconstructionBackend,
    DepthEstimationBackend, ...) returns a plain in-memory domain object and
    lets the pipeline layer decide how to serialize/cache it. This one takes
    an output path and writes the file itself, because a render engine like
    Blender runs as an external subprocess - there's no cheap way to hand
    an in-memory image back across that boundary. Don't copy this shape for
    a port that doesn't have the same constraint.
    """

    name: str

    def render(self, cloud: GaussianCloud, output_path: Path, **params) -> None: ...
