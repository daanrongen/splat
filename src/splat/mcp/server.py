"""Assembles the `splat mcp` server, one tool per CLI command.
Every remote-capable tool goes through registry.wiring.get_client(), same as
cli/*.py, so SPLAT_URL transparently redirects tool calls to a remote
`splat http` server exactly like the CLI. `splat mcp` itself is still always
a local stdio process; only where its tools execute can be remote.
"""

import functools
from collections.abc import Callable
from importlib.metadata import version
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from splat.mcp import caption as caption_tool
from splat.mcp import depth as depth_tool
from splat.mcp import diffuse, gaussian
from splat.mcp import embed as embed_tool
from splat.mcp import export as export_tool
from splat.mcp import inspect as inspect_tool
from splat.mcp import manifest as manifest_tool
from splat.mcp import mesh as mesh_tool
from splat.mcp import models as models_tool
from splat.mcp import render as render_tool
from splat.mcp import segment as segment_tool
from splat.mcp import upscale as upscale_tool

server = MCPServer("splat", version=version("splat"))


def _as_tool_error(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Any exception -> ToolError, so a failure reaches the client with its
    message as an is_error result rather than a bare 'Error executing tool'."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


server.add_tool(_as_tool_error(diffuse.diffuse), name="diffuse")
server.add_tool(_as_tool_error(caption_tool.caption), name="caption")
server.add_tool(_as_tool_error(embed_tool.embed), name="embed")
server.add_tool(_as_tool_error(segment_tool.segment), name="segment")
server.add_tool(_as_tool_error(depth_tool.depth), name="depth")
server.add_tool(_as_tool_error(upscale_tool.upscale), name="upscale")
server.add_tool(_as_tool_error(gaussian.gaussian), name="gaussian")
server.add_tool(_as_tool_error(render_tool.render), name="render")
server.add_tool(_as_tool_error(mesh_tool.mesh), name="mesh")
server.add_tool(_as_tool_error(export_tool.export), name="export")
server.add_tool(_as_tool_error(inspect_tool.info), name="info")
server.add_tool(_as_tool_error(inspect_tool.validate), name="validate")
server.add_tool(_as_tool_error(models_tool.list_models), name="models_list")
server.add_tool(_as_tool_error(models_tool.pull), name="models_pull")
server.add_tool(_as_tool_error(models_tool.info), name="models_info")
server.add_tool(_as_tool_error(models_tool.rm), name="models_rm")
server.add_tool(_as_tool_error(manifest_tool.list_manifests), name="manifest_list")
server.add_tool(_as_tool_error(manifest_tool.get), name="manifest_get")
server.add_tool(_as_tool_error(manifest_tool.delete), name="manifest_delete")


def run_stdio() -> None:
    server.run(transport="stdio")
