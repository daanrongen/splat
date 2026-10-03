"""`import splat` - a Pythonic SDK over the same handlers the CLI calls.
See `splat.api` for the wrapped functions and `splat.domain.manifest.Manifest`
for the `.as_image()`/`.as_gaussian_cloud()`/... decoding helpers.
"""

from importlib.metadata import version

__version__ = version("splat")

_API_EXPORTS = {
    "caption",
    "depth",
    "diffuse",
    "embed",
    "gaussian",
    "info",
    "render",
    "segment",
    "export",
    "mesh",
    "upscale",
    "validate",
}


def __getattr__(name: str) -> object:
    if name in _API_EXPORTS:
        from splat import api

        value = getattr(api, name)
    elif name in {"Manifest", "ManifestKind"}:
        from splat import domain
        from splat.domain import manifest

        value = getattr(manifest, name)
        _ = domain
    else:
        raise AttributeError(f"module 'splat' has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "Manifest",
    "ManifestKind",
    "__version__",
    "caption",
    "depth",
    "diffuse",
    "embed",
    "export",
    "gaussian",
    "info",
    "mesh",
    "render",
    "segment",
    "upscale",
    "validate",
]
