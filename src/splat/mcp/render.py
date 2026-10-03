import mcp.types as types

from splat.domain.manifest import ManifestKind
from splat.handlers.render import RenderRequest, handle
from splat.mcp._content import image_content, text_content
from splat.mcp._inputs import resolve_input_asset
from splat.registry.wiring import get_manifest_repository


def render(
    cloud: str,
    width: int = 1280,
    height: int = 720,
    samples: int = 32,
    engine: str = "cycles",
    background: str = "black",
    azimuth: float | None = None,
    elevation: float | None = None,
    distance: float | None = None,
    fov: float | None = None,
    look_at: str | None = None,
) -> list[types.ContentBlock]:
    """Render a Gaussian splat (path or @<asset-id>) to a PNG and return the image."""
    cache = get_manifest_repository()
    asset = resolve_input_asset(cloud, cache, default_kind=ManifestKind.GAUSSIAN_CLOUD)
    request = RenderRequest(
        inputs=[asset],
        width=width,
        height=height,
        samples=samples,
        engine=engine,
        background=background,
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        fov=fov,
        look_at=look_at,
    )
    result = handle(request)[0]
    return [text_content(f"asset id: {result.id}"), image_content(result)]
