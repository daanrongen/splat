import typer


def http(
    host: str = typer.Option("127.0.0.1", "--host", envvar="SPLAT_HTTP_BIND_HOST"),
    port: int = typer.Option(8000, "--port", envvar="SPLAT_HTTP_BIND_PORT"),
) -> None:
    """Start splat's HTTP server (no authentication — trusted-LAN use only)."""
    import uvicorn

    from splat.http.app import app

    uvicorn.run(app, host=host, port=port)
