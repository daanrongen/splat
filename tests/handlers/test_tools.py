from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from splat.adapters.cache.filesystem import FilesystemAssetCache
from splat.domain.asset import AssetKind
from splat.domain.errors import SplatDomainError
from splat.handlers.tools import DisplaceHeightRequest, handle


def _sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    Image.new("RGB", (4, 4)).save(path)
    return path


def _depth_asset(tmp_path: Path, cache: FilesystemAssetCache):
    image_asset = cache.put_external(_sample_image(tmp_path), kind=AssetKind.IMAGE)
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, np.full((4, 4), 2.0, dtype=np.float32))
    depth_asset = cache.put(
        "depthkey1",
        kind=AssetKind.DEPTH_MAP,
        content_bytes=depth_path.read_bytes(),
        ext="npy",
        metadata={"focal_length_px": 50.0, "field_of_view_deg": 30.0},
        parent_ids=[image_asset.id],
        created_by="depth:depth-pro",
    )
    return depth_asset, image_asset


def test_handle_produces_mesh_asset(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemAssetCache(tmp_path / "cache")
    depth_asset, image_asset = _depth_asset(tmp_path, cache)

    results = handle(DisplaceHeightRequest(inputs=[depth_asset]))

    assert len(results) == 1
    assert results[0].kind == AssetKind.SHAPE_3D
    assert sorted(results[0].parent_ids) == sorted([image_asset.id, depth_asset.id])


def test_handle_rejects_non_depth_input(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemAssetCache(tmp_path / "cache")
    image_asset = cache.put_external(_sample_image(tmp_path), kind=AssetKind.IMAGE)

    with pytest.raises(SplatDomainError, match="requires a depth map"):
        handle(DisplaceHeightRequest(inputs=[image_asset]))
