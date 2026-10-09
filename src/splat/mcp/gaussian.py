from pathlib import Path

import mcp.types as types

from splat.domain.manifest import ManifestKind
from splat.handlers.gaussian import GaussianRequest
from splat.mcp._content import resource_content, text_content
from splat.mcp._inputs import resolve_input_asset
from splat.registry.wiring import get_client, get_manifest_repository, get_reader, get_writer


def gaussian(
    images: list[str],
    output_path: str | None = None,
    model: str = "sharp",
    device: str = "auto",
    quality: str = "fast",
    focal_35mm: float = 30.0,
    iters: int | None = None,
    max_dim: int | None = None,
    sh_degree: int | None = None,
    poses: str = "auto",
    refine_poses: str = "auto",
    low_memory: bool = False,
    seed: int = 0,
    declutter: bool = False,
    normalize_color: bool = True,
    orbit_frames: int | None = None,
    orbit_degrees: float = 30.0,
    score: bool = False,
    mask: str | None = None,
) -> list[types.ContentBlock]:
    """Reconstruct a Gaussian splat from image paths or @asset ids."""
    cache = get_manifest_repository()
    inputs = [resolve_input_asset(p, cache, default_kind=ManifestKind.IMAGE) for p in images]
    result = get_client().gaussian(
        GaussianRequest(
            inputs=inputs,
            model=model,
            device=device,
            quality=quality,
            focal_35mm=focal_35mm,
            iters=iters,
            max_dim=max_dim,
            sh_degree=sh_degree,
            poses=poses,
            refine_poses=refine_poses,
            low_memory=low_memory,
            seed=seed,
            declutter=declutter,
            normalize_color=normalize_color,
            orbit_frames=orbit_frames,
            orbit_degrees=orbit_degrees,
            score=score,
            mask=resolve_input_asset(mask, cache, default_kind=ManifestKind.STICKER)
            if mask
            else None,
        )
    )[0]
    content: list[types.ContentBlock] = [
        text_content(f"asset id: {result.id}"),
        resource_content(result),
    ]
    if output_path is not None:
        output = Path(output_path)
        cloud = get_reader(result.content_path.suffix).read(result.content_path)
        writer = get_writer(output.suffix)
        warnings = writer.supports(cloud)
        writer.write(cloud, output)
        content.insert(
            0,
            text_content(f"wrote {output_path} ({result.metadata.point_count:,} points)"),
        )
        content.extend(text_content(f"warning: {w}") for w in warnings)
    return content
