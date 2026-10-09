import os

import typer

from splat.cli._console import warn


def http(
    host: str = typer.Option("127.0.0.1:8000", "--host", envvar="SPLAT_HOST"),
) -> None:
    """Start splat's HTTP server (set SPLAT_TOKEN to require a bearer token)."""
    import uvicorn

    from splat.http.app import app

    bind_host, _, bind_port = host.rpartition(":")
    if bind_host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("SPLAT_TOKEN"):
        warn("non-loopback bind without SPLAT_TOKEN: anyone who can reach it can run models")
    uvicorn.run(app, host=bind_host or host, port=int(bind_port) if bind_port else 8000)
