def mcp() -> None:
    """Start splat's MCP server (stdio transport; always runs locally)."""
    from splat.mcp.server import run_stdio

    run_stdio()
