"""The universal input/output contract shared by every generative-pipeline
command (diffuse/segment/depth/...): resolve INPUT from a file path,
`@<asset-id>`, or piped NDJSON asset records; report OUTPUT as NDJSON when
piped, or a human summary in an interactive terminal.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path

from splat.domain.asset import Asset, AssetKind
from splat.domain.errors import SplatDomainError
from splat.ports.asset_cache import AssetCache


def is_piped() -> bool:
    return not sys.stdout.isatty()


def resolve_inputs(
    input_arg: str | None, cache: AssetCache, *, default_kind: AssetKind = AssetKind.IMAGE
) -> list[Asset]:
    if input_arg in (None, "-"):
        if sys.stdin.isatty():
            raise SplatDomainError(
                "No input given and stdin isn't piped. Provide a file path, @<asset-id>, "
                "or pipe NDJSON asset records from another splat command."
            )
        assets = []
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                continue
            record = json.loads(line)
            assets.append(cache.get(record["id"]))
        if not assets:
            raise SplatDomainError("No asset records received on stdin.")
        return assets

    if input_arg.startswith("@"):
        return [cache.get(input_arg[1:])]

    path = Path(input_arg)
    if not path.exists():
        raise SplatDomainError(f"Input path {input_arg!r} does not exist.")
    return [cache.put_external(path, kind=default_kind)]


def report(assets: list[Asset], human: Callable[[list[Asset]], None]) -> None:
    if is_piped():
        for asset in assets:
            print(
                json.dumps(
                    {
                        "id": asset.id,
                        "kind": asset.kind.value,
                        "path": str(asset.content_path),
                        "metadata": asset.metadata,
                        "parent_ids": asset.parent_ids,
                        "created_by": asset.created_by,
                    }
                )
            )
    else:
        human(assets)
