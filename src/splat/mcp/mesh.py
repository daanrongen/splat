import mcp.types as types

from splat.domain.manifest import ManifestKind
from splat.handlers.mesh import MeshRequest, handle
from splat.mcp._content import resource_content, text_content
from splat.mcp._inputs import resolve_input_asset
from splat.registry.wiring import get_manifest_repository


def mesh(
    input: str,
    model: str | None = None,
    format: str = "glb",
    depth: int = 9,
    opacity_threshold: float = 0.1,
) -> list[types.ContentBlock]:
    """Mesh a metric depth map (heightfield) or a Gaussian cloud (poisson), given a path
    or @<asset-id>; the model follows the input kind unless set."""
    cache = get_manifest_repository()
    asset = resolve_input_asset(input, cache, default_kind=ManifestKind.GAUSSIAN_CLOUD)
    request = MeshRequest(
        inputs=[asset],
        model=model,
        format=format,
        depth=depth,
        opacity_threshold=opacity_threshold,
    )
    result = handle(request)[0]
    extra = result.metadata.extra
    return [
        text_content(
            f"asset id: {result.id} ({extra.get('vertex_count', 0):,} vertices, "
            f"{extra.get('face_count', 0):,} faces)"
        ),
        resource_content(result),
    ]
