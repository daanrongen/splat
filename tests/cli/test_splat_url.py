"""End-to-end: SPLAT_URL redirects a CLI command to a live `splat http`
server, matching the target scenario (one machine runs `splat http`,
another drives it with SPLAT_URL set)."""

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from typer.testing import CliRunner

from splat.cli.main import app
from splat.domain.value_objects import MIT
from splat.http.app import app as http_app

runner = CliRunner()


class FakeDiffusionBackend:
    name = "fake-diffuser"
    license = MIT

    def diffuse(self, prompt, *, output_path: Path, **params) -> Path:
        output_path.write_bytes(b"fake-png-bytes")
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


def test_splat_url_redirects_diffuse_to_remote_server(
    mocker, tmp_path, monkeypatch, live_server_url
):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend", return_value=FakeDiffusionBackend()
    )
    out_path = tmp_path / "test.png"

    result = runner.invoke(
        app,
        ["diffuse", "a fox", "--model", "fake-diffuser", "-o", str(out_path)],
        env={"SPLAT_URL": live_server_url},
    )

    assert result.exit_code == 0, result.output
    assert out_path.read_bytes() == b"fake-png-bytes"
