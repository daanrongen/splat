from importlib.metadata import version

import numpy as np

from splat.adapters.formats.image import encode_png
from splat.adapters.formats.ply import PlyWriter
from splat.mcp.server import server

_FAKE_PNG = encode_png(np.zeros((4, 8, 3), dtype=np.uint8))


class FakeRenderBackend:
    name = "fake-render"

    def render(self, cloud, output_path, **params):
        output_path.write_bytes(_FAKE_PNG)


def test_render_returns_the_image_so_the_agent_can_see_it(
    mocker, tmp_path, synthetic_cloud, call_tool
):
    mocker.patch.dict("os.environ", {"SPLAT_MANIFEST_CACHE_DIR": str(tmp_path / "cache")})
    mocker.patch("splat.handlers.render.get_render_backend", return_value=FakeRenderBackend())
    ply_path = tmp_path / "cloud.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    result = call_tool("render", cloud=str(ply_path), azimuth=90.0)

    assert result.is_error is False
    assert [block.type for block in result.content] == ["text", "image"]
    assert result.content[1].mime_type == "image/png"


def test_server_version_matches_the_package():
    assert server.version == version("splat")
