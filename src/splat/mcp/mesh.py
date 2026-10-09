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
    resolution: int = 192,
    opacity_threshold: float = 0.1,
) -> list[types.ContentBlock]:
    """Mesh a metric depth map (heightfield) or a Gaussian cloud (isosurface), given a path
    or @<asset-id>; the model follows the input kind unless set."""
    cache = get_manifest_repository()
    asset = resolve_input_asset(input, cache, default_kind=ManifestKind.GAUSSIAN_CLOUD)
    request = MeshRequest(
        inputs=[asset],
        model=model,
        format=format,
        resolution=resolution,
        opacity_threshold=opacity_threshold,
    )
    result = handle(request)[0]
    mesh = result.metadata
    return [
        text_content(
            f"asset id: {result.id} ({mesh.vertex_count:,} vertices, {mesh.face_count:,} faces)"
        ),
        resource_content(result),
    ]
