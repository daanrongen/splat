"""The Python SDK: `import splat; manifest = splat.diffuse("a fox")`.

Thin, Pythonic wrappers over handlers/*.py — the same request/response
shapes the CLI uses, minus Typer's argument parsing and console/exit-code
error handling (domain errors raise directly). Every function here routes
through `registry.wiring.get_client()`, exactly like the CLI, so setting
`SPLAT_URL` redirects remote-capable calls transparently — the one exception
are `render()`, `mesh()` and `export()`, which (like the CLI) always execute
locally; see `ports/client.py`'s docstring for why.
"""

from pathlib import Path

from splat.domain.manifest import Manifest, ManifestKind
from splat.handlers.caption import DEFAULT_CAPTION_PROMPT, CaptionRequest
from splat.handlers.depth import DepthRequest
from splat.handlers.diffuse import DiffuseRequest, DiffuseResult
from splat.handlers.embed import EmbedRequest
from splat.handlers.export import ExportResult
from splat.handlers.gaussian import GaussianRequest
from splat.handlers.segment import SegmentRequest
from splat.handlers.upscale import UpscaleRequest
from splat.ports.client import InfoSummary, ValidationSummary
from splat.registry.wiring import get_client, get_manifest_repository

ManifestLike = str | Path | Manifest


def _resolve(item: ManifestLike, *, default_kind: ManifestKind) -> Manifest:
    if isinstance(item, Manifest):
        return item
    text = str(item)
    if text.startswith("@"):
        return Manifest.load(text)
    return get_manifest_repository().put_external(Path(text), kind=default_kind)


def _resolve_all(
    items: ManifestLike | list[ManifestLike], *, default_kind: ManifestKind = ManifestKind.IMAGE
) -> list[Manifest]:
    if isinstance(items, (str, Path, Manifest)):
        items = [items]
    return [_resolve(item, default_kind=default_kind) for item in items]


def diffuse(
    prompt: str,
    image: ManifestLike | None = None,
    *,
    model: str = "sdxl-turbo-mlx",
    negative_prompt: str = "",
    steps: int | None = None,
    strength: float | None = None,
    seed: int | None = None,
    device: str = "auto",
) -> DiffuseResult:
    return get_client().diffuse(
        DiffuseRequest(
            prompt=prompt,
            inputs=_resolve_all(image) if image is not None else [],
            model=model,
            negative_prompt=negative_prompt,
            steps=steps,
            strength=strength,
            seed=seed,
            device=device,
        )
    )


def caption(
    inputs: ManifestLike | list[ManifestLike],
    *,
    model: str = "fastvlm-0.5b",
    prompt: str = DEFAULT_CAPTION_PROMPT,
    max_tokens: int = 80,
    temperature: float = 0.0,
    device: str = "auto",
) -> list[Manifest]:
    return get_client().caption(
        CaptionRequest(
            inputs=_resolve_all(inputs),
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            device=device,
        )
    )


def segment(
    inputs: ManifestLike | list[ManifestLike],
    *,
    model: str = "sam-mlx",
    max_stickers: int = 20,
    device: str = "auto",
) -> list[Manifest]:
    return get_client().segment(
        SegmentRequest(
            inputs=_resolve_all(inputs), model=model, max_stickers=max_stickers, device=device
        )
    )


def depth(
    inputs: ManifestLike | list[ManifestLike], *, model: str = "depth-pro", device: str = "auto"
) -> list[Manifest]:
    return get_client().depth(DepthRequest(inputs=_resolve_all(inputs), model=model, device=device))


def upscale(
    inputs: ManifestLike | list[ManifestLike],
    *,
    model: str = "realesrgan-mlx",
    factor: int = 4,
    tile: int = 0,
) -> list[Manifest]:
    return get_client().upscale(
        UpscaleRequest(inputs=_resolve_all(inputs), model=model, factor=factor, tile=tile)
    )


def embed(
    inputs: ManifestLike | list[ManifestLike] | None = None,
    *,
    text: str | None = None,
    model: str = "mobileclip2-s0",
    device: str = "auto",
) -> list[Manifest]:
    resolved = _resolve_all(inputs) if inputs is not None else None
    return get_client().embed(EmbedRequest(inputs=resolved, text=text, model=model, device=device))


def gaussian(
    inputs: ManifestLike | list[ManifestLike],
    *,
    model: str = "sharp",
    device: str = "auto",
    quality: str = "fast",
    iters: int | None = None,
    max_dim: int | None = None,
    sh_degree: int | None = None,
    poses: str = "auto",
    refine_poses: str = "auto",
    low_memory: bool = False,
    seed: int = 0,
    score: bool = False,
) -> list[Manifest]:
    return get_client().gaussian(
        GaussianRequest(
            inputs=_resolve_all(inputs),
            model=model,
            device=device,
            quality=quality,
            iters=iters,
            max_dim=max_dim,
            sh_degree=sh_degree,
            poses=poses,
            refine_poses=refine_poses,
            low_memory=low_memory,
            seed=seed,
            score=score,
        )
    )


def render(
    inputs: ManifestLike | list[ManifestLike],
    *,
    model: str = "blender",
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
) -> list[Manifest]:
    from splat.handlers.render import RenderRequest
    from splat.handlers.render import handle as handle_render

    resolved = _resolve_all(inputs, default_kind=ManifestKind.GAUSSIAN_CLOUD)
    return handle_render(
        RenderRequest(
            inputs=resolved,
            model=model,
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
    )


def mesh(
    inputs: ManifestLike | list[ManifestLike],
    *,
    model: str | None = None,
    format: str = "glb",
    depth: int = 8,
    opacity_threshold: float = 0.1,
) -> list[Manifest]:
    from splat.handlers.mesh import MeshRequest
    from splat.handlers.mesh import handle as handle_mesh

    resolved = _resolve_all(inputs, default_kind=ManifestKind.GAUSSIAN_CLOUD)
    return handle_mesh(
        MeshRequest(
            inputs=resolved,
            model=model,
            format=format,
            depth=depth,
            opacity_threshold=opacity_threshold,
        )
    )


def export(
    input: ManifestLike,
    output_path: str | Path,
    *,
    profile: str | None = None,
    pruning: str = "threshold",
    target_count: int | None = None,
) -> ExportResult:
    from splat.handlers.export import ExportRequest
    from splat.handlers.export import handle as handle_export

    return handle_export(
        ExportRequest(
            input=_resolve(input, default_kind=ManifestKind.GAUSSIAN_CLOUD),
            output_path=Path(output_path),
            profile=profile,
            pruning=pruning,
            target_count=target_count,
        )
    )


def info(path: str | Path) -> InfoSummary:
    return get_client().info(Path(path))


def validate(path: str | Path, *, strict: bool = False) -> ValidationSummary:
    return get_client().validate(Path(path), strict=strict)
