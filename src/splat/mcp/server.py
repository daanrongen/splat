"""Assembles the `splat mcp` server — the MCP driving adapter, mirroring
cli/main.py's per-command registration. Every tool calls handlers/*.py
directly (never registry/application/adapters directly), and always
executes locally: this server never consults SPLAT_URL.
"""

import functools
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from splat.domain.errors import SplatDomainError
from splat.mcp import compress, convert, diffuse, gaussian
from splat.mcp import depth as depth_tool
from splat.mcp import inspect as inspect_tool
from splat.mcp import mesh as mesh_tool
from splat.mcp import models as models_tool
from splat.mcp import segment as segment_tool
from splat.mcp import tools as tools_tool

server = MCPServer("splat", version="0.1.0")


def _as_tool_error(fn: Callable[..., Any]) -> Callable[..., Any]:
    """SplatDomainError -> ToolError, so a domain failure reaches the client
    as a graceful is_error result rather than crashing the tool call."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except SplatDomainError as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


server.add_tool(_as_tool_error(diffuse.diffuse), name="diffuse")
server.add_tool(_as_tool_error(segment_tool.segment), name="segment")
server.add_tool(_as_tool_error(depth_tool.depth), name="depth")
server.add_tool(_as_tool_error(mesh_tool.mesh), name="mesh")
server.add_tool(_as_tool_error(gaussian.gaussian), name="gaussian")
server.add_tool(_as_tool_error(convert.convert), name="convert")
server.add_tool(_as_tool_error(compress.compress), name="compress")
server.add_tool(_as_tool_error(inspect_tool.info), name="info")
server.add_tool(_as_tool_error(inspect_tool.validate), name="validate")
server.add_tool(_as_tool_error(models_tool.list_models), name="models_list")
server.add_tool(_as_tool_error(models_tool.pull), name="models_pull")
server.add_tool(_as_tool_error(models_tool.info), name="models_info")
server.add_tool(_as_tool_error(models_tool.rm), name="models_rm")
server.add_tool(_as_tool_error(tools_tool.displace_height), name="tools_displace_height")


def run_stdio() -> None:
    server.run(transport="stdio")
