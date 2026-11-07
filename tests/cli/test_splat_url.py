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

from splat.adapters.formats.image import read_rgb_or_rgba
from splat.adapters.formats.ply import PlyWriter
from splat.cli.main import app
from splat.domain.value_objects import MIT
from splat.http.app import app as http_app
from tests.image_helpers import sample_png_bytes, write_sample_png

runner = CliRunner()


class FakeDiffusionBackend:
    name = "fake-diffuser"
    license = MIT

    def diffuse(self, prompt, *, output_path: Path, **params) -> Path:
        output_path.write_bytes(sample_png_bytes())
        return output_path


class FakeRenderBackend:
    name = "fake-render"

    def __init__(self) -> None:
        self.calls = 0

    def render(self, cloud, output_path: Path, **params) -> None:
        self.calls += 1
        output_path.write_bytes(sample_png_bytes((6, 4)))


class FakeUpscaleBackend:
    name = "fake-upscaler"
    license = MIT
    supported_factors = (2, 4)

    def upscale(self, image, *, factor: int, tile: int = 0, **params):
        import numpy as np

        return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


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
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
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
    assert out_path.read_bytes() == sample_png_bytes()


def test_splat_url_redirects_upscale_to_remote_server(
    mocker, tmp_path, monkeypatch, live_server_url
):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.upscale.get_upscale_backend", return_value=FakeUpscaleBackend())
    input_path = write_sample_png(tmp_path / "input.png", (3, 2))
    out_path = tmp_path / "upscaled.png"

    result = runner.invoke(
        app,
        [
            "upscale",
            str(input_path),
            "--model",
            "fake-upscaler",
            "--factor",
            "2",
            "-o",
            str(out_path),
        ],
        env={"SPLAT_URL": live_server_url},
    )

    assert result.exit_code == 0, result.output
    assert read_rgb_or_rgba(out_path).shape == (4, 6, 3)


def test_splat_url_renders_on_the_server_not_the_client(
    mocker, tmp_path, monkeypatch, live_server_url, synthetic_cloud
):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    backend = FakeRenderBackend()
    mocker.patch("splat.handlers.render.get_render_backend", return_value=backend)
    local_render = mocker.patch("splat.adapters.client.local.LocalSplatClient.render")
    cloud_path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, cloud_path)
    out_path = tmp_path / "render.png"

    result = runner.invoke(
        app,
        ["render", str(cloud_path), "--model", "fake-render", "-o", str(out_path)],
        env={"SPLAT_URL": live_server_url},
    )

    assert result.exit_code == 0, result.output
    assert backend.calls == 1  # the server's backend ran
    local_render.assert_not_called()  # the client's own did not
    assert read_rgb_or_rgba(out_path).shape == (4, 6, 3)


def test_local_flag_renders_here_even_when_splat_url_is_set(
    mocker, tmp_path, monkeypatch, live_server_url, synthetic_cloud
):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch.dict("os.environ")  # --local sets SPLAT_LOCAL; restore it afterwards
    server_backend = FakeRenderBackend()
    mocker.patch("splat.handlers.render.get_render_backend", return_value=server_backend)
    local_render = mocker.patch(
        "splat.adapters.client.local.LocalSplatClient.render", return_value=[]
    )
    cloud_path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, cloud_path)

    runner.invoke(
        app,
        ["--local", "render", str(cloud_path), "--model", "fake-render"],
        env={"SPLAT_URL": live_server_url},
    )

    local_render.assert_called_once()
    assert server_backend.calls == 0


def test_splat_url_meshes_and_exports_on_the_server(
    mocker, tmp_path, monkeypatch, live_server_url, synthetic_cloud
):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    local_mesh = mocker.patch("splat.adapters.client.local.LocalSplatClient.mesh")
    local_export = mocker.patch("splat.adapters.client.local.LocalSplatClient.export")
    cloud_path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, cloud_path)
    env = {"SPLAT_URL": live_server_url}

    mesh_path = tmp_path / "mesh.glb"
    meshed = runner.invoke(
        app,
        [
            "mesh",
            str(cloud_path),
            "--resolution",
            "24",
            "--opacity-threshold",
            "0",
            "-o",
            str(mesh_path),
        ],
        env=env,
    )
    exported = runner.invoke(
        app, ["export", str(cloud_path), "-o", str(tmp_path / "cloud.splat")], env=env
    )

    assert meshed.exit_code == 0, meshed.output
    assert exported.exit_code == 0, exported.output
    assert mesh_path.stat().st_size > 0
    assert (tmp_path / "cloud.splat").stat().st_size > 0
    local_mesh.assert_not_called()
    local_export.assert_not_called()
