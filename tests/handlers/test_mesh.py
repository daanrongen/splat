import json
from pathlib import Path

import numpy as np
import pytest

from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.adapters.formats.ply import PlyWriter
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import DepthMetadata
from splat.handlers.mesh import MeshRequest, handle
from tests.adapters.mesh.test_isosurface import _sphere_cloud
from tests.image_helpers import depth_npy, write_sample_png


def _sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    return write_sample_png(path, (4, 4))


def _depth_asset(tmp_path: Path, cache: FilesystemManifestRepository):
    image_asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, np.full((4, 4), 2.0, dtype=np.float32))
    depth_asset = cache.put(
        "depthkey1",
        kind=ManifestKind.DEPTH_MAP,
        content_bytes=depth_path.read_bytes(),
        ext="npy",
        metadata=DepthMetadata(focal_length_px=50.0, field_of_view_deg=30.0),
        parent_ids=[image_asset.id],
        created_by="depth:depth-pro",
    )
    return depth_asset, image_asset


def test_handle_produces_mesh_asset(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemManifestRepository(tmp_path / "cache")
    depth_asset, image_asset = _depth_asset(tmp_path, cache)

    results = handle(MeshRequest(inputs=[depth_asset]))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.SHAPE_3D
    assert results[0].created_by == "mesh:heightfield"
    assert sorted(results[0].parent_ids) == sorted([image_asset.id, depth_asset.id])


def test_handle_rejects_inputs_it_cannot_mesh(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemManifestRepository(tmp_path / "cache")
    image_asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    with pytest.raises(SplatDomainError, match="needs a depth_map"):
        handle(MeshRequest(inputs=[image_asset]))


def test_handle_rejects_relative_disparity_depth(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemManifestRepository(tmp_path / "cache")
    image_asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)
    np.save(tmp_path / "disparity.npy", np.full((4, 4), 2.0, dtype=np.float32))
    disparity = cache.put(
        "disparitykey",
        kind=ManifestKind.DEPTH_MAP,
        content_bytes=(tmp_path / "disparity.npy").read_bytes(),
        ext="npy",
        metadata=DepthMetadata(focal_length_px=50.0, units="disparity"),
        parent_ids=[image_asset.id],
        created_by="depth:depth-anything-v2-coreml",
    )

    with pytest.raises(SplatDomainError, match="relative disparity"):
        handle(MeshRequest(inputs=[disparity]))


def test_legacy_relative_depth_manifest_reads_as_disparity(tmp_path: Path):
    cache = FilesystemManifestRepository(tmp_path / "cache")
    cache.put(
        "legacykey",
        kind=ManifestKind.DEPTH_MAP,
        content_bytes=depth_npy(),
        ext="npy",
        metadata=DepthMetadata(extra={"relative": True}),
        parent_ids=[],
        created_by="depth:depth-anything-v2-coreml",
    )
    meta_path = tmp_path / "cache" / "legacykey.meta.json"
    raw = json.loads(meta_path.read_text())
    del raw["metadata"]["units"]
    meta_path.write_text(json.dumps(raw))

    assert cache.get("legacykey").metadata.units == "disparity"


def test_handle_meshes_a_gaussian_cloud_with_isosurface(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemManifestRepository(tmp_path / "cache")
    PlyWriter().write(_sphere_cloud(), tmp_path / "sphere.ply")
    cloud = cache.put_external(tmp_path / "sphere.ply", kind=ManifestKind.GAUSSIAN_CLOUD)

    result = handle(MeshRequest(inputs=[cloud], format="obj", resolution=48))[0]

    assert result.created_by == "mesh:isosurface"
    assert result.parent_ids == [cloud.id]
    assert result.metadata.face_count > 0
    assert result.content_path.read_bytes().startswith(b"#")


def test_isosurface_rejects_a_depth_map(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    depth_asset, _ = _depth_asset(tmp_path, FilesystemManifestRepository(tmp_path / "cache"))

    with pytest.raises(SplatDomainError, match="isosurface needs a gaussian_cloud"):
        handle(MeshRequest(inputs=[depth_asset], model="isosurface"))
