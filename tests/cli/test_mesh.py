import json
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from splat.cli.main import app
from splat.domain.image_space import Shape3D
from splat.domain.value_objects import MIT

runner = CliRunner()


class FakeMeshBackend:
    name = "triposr"
    license = MIT

    def predict(self, image_path, **params) -> Shape3D:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        return Shape3D(vertices=vertices, faces=faces, metadata={"face_count": 2})


def _sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    Image.new("RGB", (4, 4)).save(path)
    return path


def test_mesh_predicts_with_wired_backend(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.cli.mesh.get_mesh_backend", return_value=FakeMeshBackend())
    image_path = _sample_image(tmp_path)

    result = runner.invoke(app, ["mesh", str(image_path)])

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["kind"] == "shape_3d"
    assert line["created_by"] == "mesh:triposr"


def test_mesh_unknown_model_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    image_path = _sample_image(tmp_path)

    result = runner.invoke(app, ["mesh", str(image_path), "--model", "nope"])

    assert result.exit_code == 1
    assert "Unknown mesh model" in result.output


def test_mesh_default_model_reports_not_implemented(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    image_path = _sample_image(tmp_path)

    result = runner.invoke(app, ["mesh", str(image_path)])

    assert result.exit_code == 1
    assert "not yet implemented" in result.output
