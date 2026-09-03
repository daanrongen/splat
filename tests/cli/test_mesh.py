import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from splat.adapters.cache.filesystem import FilesystemAssetCache
from splat.cli.main import app
from splat.domain.asset import AssetKind
from splat.domain.image_space import DepthMap, Shape3D
from splat.domain.value_objects import APPLE_ASCL, MIT

runner = CliRunner()


class FakeDepthBackend:
    name = "depth-pro"
    license = APPLE_ASCL

    def estimate(self, image_path, **params) -> DepthMap:
        depth = np.full((4, 4), 2.0, dtype=np.float32)
        return DepthMap(
            depth=depth,
            focal_length_px=100.0,
            field_of_view_deg=30.0,
            metadata={"source_model": "fake/depth", "device": "cpu"},
        )


class FakeMeshBackend:
    name = "depth-heightfield"
    license = MIT

    def predict(self, image_path, *, depth_map=None, **params) -> Shape3D:
        assert depth_map is not None
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        return Shape3D(vertices=vertices, faces=faces, metadata={"face_count": 2})


def _sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    Image.new("RGB", (4, 4)).save(path)
    return path


def test_mesh_from_bare_image_auto_computes_depth(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.cli.mesh.get_mesh_backend", return_value=FakeMeshBackend())
    mocker.patch("splat.cli.mesh.get_depth_backend", return_value=FakeDepthBackend())
    image_path = _sample_image(tmp_path)

    result = runner.invoke(app, ["mesh", str(image_path)])

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["kind"] == "shape_3d"
    assert line["created_by"] == "mesh:depth-heightfield"


def test_mesh_human_output(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.cli.mesh.get_mesh_backend", return_value=FakeMeshBackend())
    mocker.patch("splat.cli.mesh.get_depth_backend", return_value=FakeDepthBackend())
    mocker.patch("splat.cli._pipeline_io.is_piped", return_value=False)
    image_path = _sample_image(tmp_path)

    result = runner.invoke(app, ["mesh", str(image_path)])
    plain_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)

    assert result.exit_code == 0, result.output
    assert "mesh" in plain_output
    assert "faces=2" in plain_output


def test_mesh_output_flag_writes_file(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.cli.mesh.get_mesh_backend", return_value=FakeMeshBackend())
    mocker.patch("splat.cli.mesh.get_depth_backend", return_value=FakeDepthBackend())
    image_path = _sample_image(tmp_path)
    out_path = tmp_path / "mesh.glb"

    result = runner.invoke(app, ["mesh", str(image_path), "-o", str(out_path)])

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_mesh_from_piped_depth_asset_reuses_it(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mock_depth_backend = mocker.patch("splat.cli.mesh.get_depth_backend")
    mocker.patch("splat.cli.mesh.get_mesh_backend", return_value=FakeMeshBackend())
    cache = FilesystemAssetCache(tmp_path / "cache")
    image_asset = cache.put_external(_sample_image(tmp_path), kind=AssetKind.IMAGE)
    depth_bytes = np.full((4, 4), 2.0, dtype=np.float32)
    buf_path = tmp_path / "depth.npy"
    np.save(buf_path, depth_bytes)
    depth_asset = cache.put(
        "depthkey1",
        kind=AssetKind.DEPTH_MAP,
        content_bytes=buf_path.read_bytes(),
        ext="npy",
        metadata={"focal_length_px": 100.0, "field_of_view_deg": 30.0},
        parent_ids=[image_asset.id],
        created_by="depth:depth-pro",
    )
    stdin_payload = json.dumps({"id": depth_asset.id}) + "\n"

    result = runner.invoke(app, ["mesh", "-"], input=stdin_payload)

    assert result.exit_code == 0, result.output
    mock_depth_backend.assert_not_called()
    line = json.loads(result.output.strip().splitlines()[-1])
    assert sorted(line["parent_ids"]) == sorted([image_asset.id, depth_asset.id])


def test_mesh_unknown_model_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    image_path = _sample_image(tmp_path)

    result = runner.invoke(app, ["mesh", str(image_path), "--model", "nope"])

    assert result.exit_code == 1
    assert "Unknown mesh model" in result.output


def test_mesh_splat_input_reports_not_implemented(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["mesh", str(tmp_path / "scene.ply"), "-o", str(tmp_path / "out.obj")]
    )

    assert result.exit_code == 1
    assert "not yet implemented" in result.output
