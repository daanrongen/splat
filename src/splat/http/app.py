"""Assembles the `splat http` server — the HTTP driving adapter, mirroring
cli/main.py's per-command router assembly. Every route calls handlers/*.py
directly (never registry/application/adapters directly), and always
executes locally: this server never consults SPLAT_URL.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from splat.domain.errors import SplatDomainError
from splat.http import (
    assets,
    caption,
    depth,
    diffuse,
    embed,
    export,
    gaussian,
    info,
    manifest,
    mesh,
    models,
    render,
    segment,
    upscale,
    validate,
)

_ROUTERS = (
    diffuse,
    caption,
    embed,
    segment,
    depth,
    upscale,
    gaussian,
    render,
    mesh,
    export,
    info,
    validate,
    models,
    assets,
    manifest,
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="splat",
        summary="splat's pipeline over HTTP. No authentication in v1 — trusted-LAN use only.",
    )

    @app.exception_handler(SplatDomainError)
    async def _handle_domain_error(request: Request, exc: SplatDomainError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _handle_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    for router_module in _ROUTERS:
        app.include_router(router_module.router)

    return app


app = create_app()
