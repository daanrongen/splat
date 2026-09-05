"""Resolving a tool argument (a local file path or `@<asset-id>`) into a
cached Manifest — the MCP analogue of cli/_pipeline_io.py's resolve_inputs,
minus the stdin/NDJSON piping that has no MCP equivalent.
"""

from pathlib import Path

from splat.domain.manifest import Manifest, ManifestKind
from splat.ports.manifest_repository import ManifestRepository


def resolve_input_asset(
    path_or_ref: str, cache: ManifestRepository, *, default_kind: ManifestKind = ManifestKind.IMAGE
) -> Manifest:
    if path_or_ref.startswith("@"):
        return cache.get(path_or_ref[1:])
    return cache.put_external(Path(path_or_ref), kind=default_kind)
