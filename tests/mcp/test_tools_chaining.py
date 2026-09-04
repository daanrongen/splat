"""Verifies that an asset id returned by one tool call can be fed as
`@<asset-id>` into another — the MCP equivalent of the CLI's `@<id>` and
Unix-pipe chaining.
"""

import numpy as np
from PIL import Image

from splat.domain.image_space import DepthMap
from splat.domain.value_objects import APPLE_ASCL


class FakeDepthBackend:
    name = "depth-pro"
    license = APPLE_ASCL

    def estimate(self, image_path, **params) -> DepthMap:
        depth = np.full((4, 4), 2.0, dtype=np.float32)
        return DepthMap(depth=depth, focal_length_px=50.0, field_of_view_deg=30.0, metadata={})


def test_depth_asset_id_chains_into_displace_height(mocker, tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.depth.get_depth_backend", return_value=FakeDepthBackend())
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (4, 4)).save(image_path)

    depth_result = call_tool("depth", image=str(image_path))
    depth_asset_id = next(
        b.text.removeprefix("asset id: ") for b in depth_result.content if b.type == "text"
    )

    mesh_result = call_tool("tools_displace_height", depth_asset_id=depth_asset_id)

    assert mesh_result.is_error is False
    assert any(block.type == "resource" for block in mesh_result.content)


def test_segment_accepts_at_id_reference(mocker, tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    from splat.domain.asset import AssetKind
    from splat.registry.wiring import get_asset_cache

    cache = get_asset_cache()
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (2, 2)).save(image_path)
    asset = cache.put_external(image_path, kind=AssetKind.IMAGE)

    class FakeSegmentationBackend:
        name = "fake-sam"
        from splat.domain.value_objects import MIT as license

        def segment(self, image_path, *, max_stickers=None, **params):
            rgba = np.zeros((2, 2, 4), dtype=np.uint8)
            from splat.domain.image_space import Sticker

            return [Sticker(rgba=rgba, bbox=(0, 0, 2, 2), score=0.9, area=4)]

    mocker.patch(
        "splat.handlers.segment.get_segmentation_backend",
        return_value=FakeSegmentationBackend(),
    )

    result = call_tool("segment", image=f"@{asset.id}")

    assert result.is_error is False
