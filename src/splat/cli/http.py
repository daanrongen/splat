import typer


def http(
    host: str = typer.Option("127.0.0.1:8000", "--host", envvar="SPLAT_HOST"),
) -> None:
    """Start splat's HTTP server (no authentication — trusted-LAN use only)."""
    import uvicorn

    from splat.http.app import app

    bind_host, _, bind_port = host.rpartition(":")
    uvicorn.run(app, host=bind_host or host, port=int(bind_port) if bind_port else 8000)
