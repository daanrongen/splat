import asyncio

import pytest

from splat.mcp.server import server


@pytest.fixture
def call_tool():
    def _call(name: str, **arguments):
        return asyncio.run(server.call_tool(name, arguments))

    return _call
