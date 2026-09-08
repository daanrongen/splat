from pathlib import Path

from splat.adapters.client.local import LocalSplatClient
from splat.adapters.formats.ply import PlyWriter
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import MIT
from splat.handlers.caption import CaptionRequest
from splat.handlers.diffuse import DiffuseRequest
from splat.handlers.embed import EmbedRequest
from splat.handlers.tools.compress import CompressRequest
from splat.handlers.tools.convert import ConvertRequest
from splat.handlers.tools.declutter import DeclutterRequest
from splat.handlers.upscale import UpscaleRequest
from tests.image_helpers import sample_png_bytes, write_sample_png


class FakeDiffusionBackend:
    name = "fake-diffuser"
    license = MIT

    def diffuse(self, prompt, *, output_path: Path, **params) -> Path:
        output_path.write_bytes(sample_png_bytes())
        return output_path


class FakeCaptionBackend:
    name = "fake-captioner"
    license = MIT

    def caption(self, image_path: Path, *, prompt: str, max_tokens: int, temperature: float):
        return f"{prompt}: small scene"


class FakeEmbeddingBackend:
    name = "fake-embedder"
    license = MIT

    def embed_image(self, image_path: Path, **params):
        import numpy as np

        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_text(self, text: str, **params):
        import numpy as np

        return np.array([0.0, 1.0], dtype=np.float32)


def test_diffuse_delegates_to_handler(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend", return_value=FakeDiffusionBackend()
    )
    client = LocalSplatClient()

    result = client.diffuse(DiffuseRequest(prompt="a fox", model="fake-diffuser"))

    assert result.asset.content_path.read_bytes() == sample_png_bytes()
    assert result.license_warning is None


def test_caption_delegates_to_handler(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.caption.get_caption_backend", return_value=FakeCaptionBackend())
    from splat.registry.wiring import get_manifest_repository

    asset = get_manifest_repository().put_external(
        write_sample_png(tmp_path / "image.png", (2, 2)), kind=ManifestKind.IMAGE
    )
    client = LocalSplatClient()

    results = client.caption(CaptionRequest(inputs=[asset], model="fake-captioner", prompt="Look"))

    assert len(results) == 1
    assert results[0].content_path.read_text(encoding="utf-8") == "Look: small scene"


def test_embed_delegates_to_handler(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.embed.get_embedding_backend", return_value=FakeEmbeddingBackend())
    client = LocalSplatClient()

    results = client.embed(EmbedRequest(text="red chair", model="fake-embedder"))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.EMBEDDING


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


def test_tools_declutter_delegates_to_handler(tmp_path, synthetic_cloud):
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    out_path = tmp_path / "out.ply"
    client = LocalSplatClient()

    cloud = client.tools_declutter(DeclutterRequest(input_path=ply_path, output_path=out_path))

    assert out_path.exists()
    assert cloud.point_count <= synthetic_cloud.point_count


def test_upscale_delegates_to_handler(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))

    class FakeUpscaleBackend:
        name = "fake-upscaler"
        license = MIT
        supported_factors = (2, 4)

        def upscale(self, image, *, factor: int, tile: int = 0, **params):
            import numpy as np

            return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)

    mocker.patch("splat.handlers.upscale.get_upscale_backend", return_value=FakeUpscaleBackend())
    from splat.registry.wiring import get_manifest_repository

    asset = get_manifest_repository().put_external(
        write_sample_png(tmp_path / "image.png", (2, 2)), kind=ManifestKind.IMAGE
    )
    client = LocalSplatClient()

    results = client.upscale(UpscaleRequest(inputs=[asset], model="fake-upscaler", factor=2))

    assert len(results) == 1
    assert results[0].created_by == "upscale:fake-upscaler"


def test_models_list_returns_summaries():
    client = LocalSplatClient()

    rows = client.models_list()

    assert len(rows) > 0
    assert all(hasattr(r, "cached") for r in rows)
