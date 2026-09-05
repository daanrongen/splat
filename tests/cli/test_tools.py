import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.cli.main import app
from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import DepthMetadata
from tests.image_helpers import write_sample_png

runner = CliRunner()


def _sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    return write_sample_png(path, (4, 4))


def _depth_asset(tmp_path: Path, cache: FilesystemManifestRepository):
    image_asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, np.full((4, 4), 2.0, dtype=np.float32))
    return cache.put(
        "depthkey1",
        kind=ManifestKind.DEPTH_MAP,
        content_bytes=depth_path.read_bytes(),
        ext="npy",
        metadata=DepthMetadata(focal_length_px=50.0, field_of_view_deg=30.0),
        parent_ids=[image_asset.id],
        created_by="depth:depth-pro",
    ), image_asset


def test_displace_height_from_piped_depth_asset(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemManifestRepository(tmp_path / "cache")
    depth_asset, image_asset = _depth_asset(tmp_path, cache)
    stdin_payload = json.dumps({"id": depth_asset.id}) + "\n"

    result = runner.invoke(app, ["tools", "displace.height", "-"], input=stdin_payload)

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["kind"] == "shape_3d"
    assert line["created_by"] == "tools:displace.height"
    assert sorted(line["parent_ids"]) == sorted([image_asset.id, depth_asset.id])


def test_displace_height_output_flag_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemManifestRepository(tmp_path / "cache")
    depth_asset, _ = _depth_asset(tmp_path, cache)
    out_path = tmp_path / "mesh.glb"

    result = runner.invoke(
        app, ["tools", "displace.height", f"@{depth_asset.id}", "-o", str(out_path)]
    )

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_displace_height_rejects_non_depth_input(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemManifestRepository(tmp_path / "cache")
    image_asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    result = runner.invoke(app, ["tools", "displace.height", f"@{image_asset.id}"])

    assert result.exit_code == 1
    assert "requires a depth map" in result.output


def test_extract_surface_reports_not_implemented(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["tools", "extract.surface", str(tmp_path / "a.ply"), str(tmp_path / "b.obj")]
    )

    assert result.exit_code == 1
    assert "not yet implemented" in result.output
