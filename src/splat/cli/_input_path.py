from pathlib import Path

from splat.domain.manifest import Manifest


def resolve_input_path(raw: str) -> Path:
    """A file path, or `@<manifest-id>` resolved to its cached content path."""
    if raw.startswith("@"):
        return Manifest.load(raw).content_path
    return Path(raw)
