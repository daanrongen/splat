from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.cli._pipeline_io import report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.caption import DEFAULT_CAPTION_PROMPT, CaptionRequest
from splat.registry.wiring import get_asset_cache, get_client


def caption(
    input: str = typer.Argument(
        ..., help="Image path, @<asset-id>, or '-' to read piped asset records."
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Write caption text here."),
    model: str = typer.Option("fastvlm-0.5b", "--model"),
    prompt: str = typer.Option(DEFAULT_CAPTION_PROMPT, "--prompt"),
    max_tokens: int = typer.Option(80, "--max-tokens", min=1),
    temperature: float = typer.Option(0.0, "--temperature", min=0.0),
    device: str = typer.Option("auto", "--device", help="auto | cpu | mps"),
) -> None:
    """Caption image(s) as UTF-8 text assets."""
    cache = get_asset_cache()
    try:
        inputs = resolve_inputs(input, cache, default_kind=ManifestKind.IMAGE)
        results = get_client().caption(
            CaptionRequest(
                inputs=inputs,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                device=device,
            )
        )
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None and results:
        output.write_text(results[0].content_path.read_text(encoding="utf-8"), encoding="utf-8")

    def _human(assets: list) -> None:
        for asset in assets:
            text = asset.content_path.read_text(encoding="utf-8")
            console.print(f"[green]caption[/green] {asset.id}  {text}")

    report(results, _human)
