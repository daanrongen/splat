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


def test_mesh_heightfield_from_piped_depth_asset(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemManifestRepository(tmp_path / "cache")
    depth_asset, image_asset = _depth_asset(tmp_path, cache)
    stdin_payload = json.dumps({"id": depth_asset.id}) + "\n"

    result = runner.invoke(app, ["mesh", "-"], input=stdin_payload)

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["kind"] == "shape_3d"
    assert line["created_by"] == "mesh:heightfield"
    assert sorted(line["parent_ids"]) == sorted([image_asset.id, depth_asset.id])


def test_mesh_heightfield_output_flag_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemManifestRepository(tmp_path / "cache")
    depth_asset, _ = _depth_asset(tmp_path, cache)
    out_path = tmp_path / "mesh.glb"

    result = runner.invoke(app, ["mesh", f"@{depth_asset.id}", "-o", str(out_path)])

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_mesh_heightfield_rejects_non_depth_input(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemManifestRepository(tmp_path / "cache")
    image_asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    result = runner.invoke(app, ["mesh", f"@{image_asset.id}"])

    assert result.exit_code == 1
    assert "needs a depth_map" in result.output


def test_mesh_refuses_a_png_path(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))

    result = runner.invoke(app, ["mesh", str(_sample_image(tmp_path))])

    assert result.exit_code == 1
    assert "cannot read a PNG" in result.output


def test_mesh_isosurface_writes_a_real_mesh(tmp_path: Path, monkeypatch) -> None:
    from splat.adapters.formats.ply import PlyWriter
    from tests.adapters.mesh.test_isosurface import _sphere_cloud

    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    ply_path = tmp_path / "sphere.ply"
    PlyWriter().write(_sphere_cloud(), ply_path)
    out_path = tmp_path / "sphere.obj"

    result = runner.invoke(app, ["mesh", str(ply_path), "-o", str(out_path), "--resolution", "48"])

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert (tmp_path / "sphere.obj.manifest.json").exists()


def test_mesh_isosurface_rejects_unsupported_format(tmp_path: Path, monkeypatch) -> None:
    from splat.adapters.formats.ply import PlyWriter
    from tests.adapters.mesh.test_isosurface import _sphere_cloud

    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    ply_path = tmp_path / "sphere.ply"
    PlyWriter().write(_sphere_cloud(), ply_path)

    result = runner.invoke(app, ["mesh", str(ply_path), "-o", str(tmp_path / "sphere.usdz")])

    assert result.exit_code == 1
    assert "Could not export mesh" in result.output


def _cloud_file(tmp_path: Path, synthetic_cloud) -> Path:
    from splat.adapters.formats.ply import PlyWriter

    path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, path)
    return path


def test_export_converts_by_extension_and_writes_a_sidecar(tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    out = tmp_path / "out" / "cloud.splat"

    result = runner.invoke(
        app, ["export", str(_cloud_file(tmp_path, synthetic_cloud)), "-o", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert (tmp_path / "out" / "cloud.splat.manifest.json").exists()


def test_export_profile_compresses_the_cloud(tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    out = tmp_path / "web.ply"

    result = runner.invoke(
        app,
        [
            "export",
            str(_cloud_file(tmp_path, synthetic_cloud)),
            "-o",
            str(out),
            "--profile",
            "web-delivery",
        ],
    )

    assert result.exit_code == 0, result.output
    from splat.adapters.formats.ply import PlyReader

    assert PlyReader().read(out).sh_degree == 0


def test_export_pruning_without_profile_is_refused(tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    out = tmp_path / "thin.ply"

    result = runner.invoke(
        app,
        [
            "export",
            str(_cloud_file(tmp_path, synthetic_cloud)),
            "-o",
            str(out),
            "--pruning",
            "blue-noise",
            "--target-count",
            "10",
        ],
    )

    assert result.exit_code == 1
    assert "need --profile" in result.output
    assert not out.exists()


def test_export_copies_non_cloud_assets_in_their_own_format(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    image = FilesystemManifestRepository(tmp_path / "cache").put_external(
        _sample_image(tmp_path), kind=ManifestKind.IMAGE
    )

    copied = runner.invoke(app, ["export", f"@{image.id}", "-o", str(tmp_path / "copy.png")])
    refused = runner.invoke(app, ["export", f"@{image.id}", "-o", str(tmp_path / "copy.jpg")])

    assert copied.exit_code == 0, copied.output
    assert (tmp_path / "copy.png").read_bytes() == image.content_path.read_bytes()
    assert refused.exit_code == 1
    assert "exports as .png" in refused.output


def test_info_refuses_a_ply_that_is_not_a_splat(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    path = tmp_path / "mesh.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
        "property float z\nend_header\n0 0 0\n"
    )

    for command in ("info", "validate"):
        result = runner.invoke(app, [command, str(path)])
        assert result.exit_code == 1
        assert "not a Gaussian splat PLY" in result.output


def test_tools_group_is_gone():
    assert runner.invoke(app, ["tools", "--help"]).exit_code == 2
