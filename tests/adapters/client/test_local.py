from pathlib import Path

from splat.adapters.client.local import LocalSplatClient
from splat.adapters.formats.ply import PlyWriter
from splat.domain.value_objects import MIT
from splat.handlers.diffuse import DiffuseRequest
from splat.handlers.tools.compress import CompressRequest
from splat.handlers.tools.convert import ConvertRequest


class FakeDiffusionBackend:
    name = "fake-diffuser"
    license = MIT

    def diffuse(self, prompt, *, output_path: Path, **params) -> Path:
        output_path.write_bytes(b"fake-png-bytes")
        return output_path


def test_diffuse_delegates_to_handler(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend", return_value=FakeDiffusionBackend()
    )
    client = LocalSplatClient()

    result = client.diffuse(DiffuseRequest(prompt="a fox", model="fake-diffuser"))

    assert result.asset.content_path.read_bytes() == b"fake-png-bytes"
    assert result.license_warning is None


def test_info_returns_summary(tmp_path, synthetic_cloud):
    ply_path = tmp_path / "a.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    client = LocalSplatClient()

    summary = client.info(ply_path)

    assert summary.points == synthetic_cloud.point_count
    assert summary.format == "ply"


def test_validate_returns_summary(tmp_path, synthetic_cloud):
    ply_path = tmp_path / "a.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    client = LocalSplatClient()

    summary = client.validate(ply_path, strict=True)

    assert summary.points == synthetic_cloud.point_count
    assert isinstance(summary.issues, list)


def test_tools_convert_delegates_to_handler(tmp_path, synthetic_cloud):
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    splat_path = tmp_path / "out.splat"
    client = LocalSplatClient()

    result = client.tools_convert(ConvertRequest(input_path=ply_path, output_path=splat_path))

    assert splat_path.exists()
    assert result.cloud.point_count == synthetic_cloud.point_count


def test_tools_compress_delegates_to_handler(tmp_path, synthetic_cloud):
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    out_path = tmp_path / "out.ply"
    client = LocalSplatClient()

    cloud = client.tools_compress(
        CompressRequest(input_path=ply_path, output_path=out_path, profile="archival")
    )

    assert out_path.exists()
    assert cloud.point_count <= synthetic_cloud.point_count


def test_models_list_returns_summaries():
    client = LocalSplatClient()

    rows = client.models_list()

    assert len(rows) > 0
    assert all(hasattr(r, "cached") for r in rows)
