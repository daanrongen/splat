"""`import splat` - a Pythonic SDK over the same handlers the CLI calls.
See `splat.api` for the wrapped functions and `splat.domain.manifest.Manifest`
for the `.as_image()`/`.as_gaussian_cloud()`/... decoding helpers.
"""

from splat.api import (
    caption,
    depth,
    diffuse,
    embed,
    gaussian,
    info,
    render,
    segment,
    tools_compress,
    tools_convert,
    upscale,
    validate,
)
from splat.domain.manifest import Manifest, ManifestKind

__version__ = "0.1.0"

__all__ = [
    "Manifest",
    "ManifestKind",
    "__version__",
    "caption",
    "depth",
    "diffuse",
    "embed",
    "gaussian",
    "info",
    "render",
    "segment",
    "tools_compress",
    "tools_convert",
    "upscale",
    "validate",
]
