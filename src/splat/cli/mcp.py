def mcp() -> None:
    """Start splat's MCP server (stdio transport; honors SPLAT_URL like the CLI)."""
    from splat.mcp.server import run_stdio

    run_stdio()
