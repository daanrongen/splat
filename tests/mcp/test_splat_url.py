"""End-to-end: SPLAT_URL redirects an MCP tool call to a live `splat http`
server, same mechanism as tests/cli/test_splat_url.py."""

import base64
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from splat.domain.value_objects import MIT
from splat.http.app import app as http_app
from tests.image_helpers import sample_png_bytes


class FakeDiffusionBackend:
    name = "fake-diffuser"
    license = MIT

    def diffuse(self, prompt, *, output_path: Path, **params) -> Path:
        output_path.write_bytes(sample_png_bytes())
        return output_path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server_url():
    port = _free_port()
    config = uvicorn.Config(http_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


def test_diffuse_tool_redirects_to_remote_server(
    mocker, tmp_path, monkeypatch, live_server_url, call_tool
):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SPLAT_URL", live_server_url)
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend", return_value=FakeDiffusionBackend()
    )

    result = call_tool("diffuse", prompt="a fox", model="fake-diffuser")

    assert result.is_error is False
    image_block = next(block for block in result.content if block.type == "image")
    assert base64.b64decode(image_block.data) == sample_png_bytes()
