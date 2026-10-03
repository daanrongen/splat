from pathlib import Path

import typer

from splat.cli._console import console, error
from splat.cli._pipeline_io import prepare_output, report, resolve_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.handlers.embed import EmbedRequest
from splat.registry.wiring import get_client, get_manifest_repository


def embed(
    input: str | None = typer.Argument(
        None, help="Image path, @<asset-id>, or '-' to read piped asset records."
    ),
    text: str | None = typer.Option(
        None, "--text", help="Embed a text query instead of image input."
    ),
    output: Path | None = typer.Option(None, "-o", "--output", help="Also export .npy here."),
    model: str = typer.Option("mobileclip2-s0", "--model", envvar="SPLAT_EMBED_MODEL"),
    device: str = typer.Option(
        "auto", "--device", help="'auto', 'mps', or 'cpu'.", envvar="SPLAT_EMBED_DEVICE"
    ),
) -> None:
    """Embed image or text inputs as normalized vector assets."""
    cache = get_manifest_repository()
    try:
        prepare_output(output)
        if text is not None:
            if input is not None:
                raise SplatDomainError("Use either image input or --text, not both.")
            results = get_client().embed(EmbedRequest(text=text, model=model, device=device))
        else:
            inputs = resolve_inputs(input, cache, default_kind=ManifestKind.IMAGE)
            results = get_client().embed(EmbedRequest(inputs=inputs, model=model, device=device))
    except SplatDomainError as exc:
        error(str(exc))
        raise typer.Exit(code=1) from exc

    if output is not None and results:
        output.write_bytes(results[0].content_path.read_bytes())

    def _human(assets: list) -> None:
        for asset in assets:
            console.print(
                f"[green]embedded[/green] {asset.id}  "
                f"{asset.metadata.input_type} "
                f"{asset.metadata.dimension}d "
                f"{asset.metadata.dtype}"
            )

    report(results, _human)
