from pathlib import Path

import typer

from splat.cli._console import console, error, warn
from splat.cli._pipeline_io import report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.diffuse import DiffuseRequest
from splat.registry.wiring import get_client, get_manifest_repository


def diffuse(
    source: str = typer.Argument(
        ..., help="Text prompt, or an image path / @<manifest-id> to edit."
    ),
    prompt: str | None = typer.Argument(
        None, help="Edit instruction, when SOURCE is an image (image-to-image)."
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Also export to this path."),
    model: str = typer.Option("sdxl-turbo-mlx", "--model", envvar="SPLAT_DIFFUSE_MODEL"),
    negative_prompt: str = typer.Option("", "--negative", envvar="SPLAT_DIFFUSE_NEGATIVE"),
    steps: int | None = typer.Option(
        None, "--steps", help="Denoising steps (model-dependent default)."
    ),
    strength: float | None = typer.Option(
        None,
        "--strength",
        help="Image-to-image noise strength 0-1 (higher = more change). Ignored for text-to-image.",
    ),
    seed: int | None = typer.Option(None, "--seed", envvar="SPLAT_DIFFUSE_SEED"),
    device: str = typer.Option(
        "auto", "--device", help="auto | cpu | mps", envvar="SPLAT_DIFFUSE_DEVICE"
    ),
) -> None:
    """Diffuse an image from a text prompt, or edit an existing image (image-to-image)."""
    cache = get_manifest_repository()
    try:
        if prompt is None:
            inputs, edit_prompt = [], source
        else:
            inputs = resolve_inputs(source, cache, default_kind=ManifestKind.IMAGE)
            edit_prompt = prompt

        result = get_client().diffuse(
            DiffuseRequest(
                prompt=edit_prompt,
                inputs=inputs,
                model=model,
                negative_prompt=negative_prompt,
                steps=steps,
                strength=strength,
                seed=seed,
                device=device,
            )
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if result.license_warning:
        warn(result.license_warning)

    if output is not None:
        output.write_bytes(result.asset.content_path.read_bytes())

    report(
        [result.asset],
        lambda assets: console.print(
            f"[green]diffused[/green] {assets[0].id} ({assets[0].content_path})"
        ),
    )
