import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from splat.adapters.formats.image import encode_png
from splat.adapters.formats.ply import PlyWriter
from splat.cli.main import app
from splat.domain.manifest import ManifestKind

runner = CliRunner()

_FAKE_PNG = encode_png(np.zeros((4, 8, 3), dtype=np.uint8))


class FakeRenderBackend:
    name = "fake-render"

    def render(self, cloud, output_path, **params):
        output_path.write_bytes(_FAKE_PNG)


def test_render_file_input_emits_ndjson_asset(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.render.get_render_backend", return_value=FakeRenderBackend())

    ply_path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    result = runner.invoke(app, ["render", str(ply_path), "--model", "fake-render"])

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["kind"] == ManifestKind.IMAGE.value
    assert line["created_by"] == "render:fake-render"


def test_render_output_option_writes_file(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.render.get_render_backend", return_value=FakeRenderBackend())

    ply_path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    output_path: Path = tmp_path / "out.png"

    result = runner.invoke(
        app, ["render", str(ply_path), "--model", "fake-render", "-o", str(output_path)]
    )

    assert result.exit_code == 0, result.output
    assert output_path.read_bytes() == _FAKE_PNG
    sidecar = json.loads((tmp_path / "out.png.manifest.json").read_text())
    assert sidecar["created_by"] == "render:fake-render"
