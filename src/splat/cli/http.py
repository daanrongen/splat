import typer


def http(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Start splat's HTTP server (no authentication — trusted-LAN use only)."""
    import uvicorn

    from splat.http.app import app

    uvicorn.run(app, host=host, port=port)
