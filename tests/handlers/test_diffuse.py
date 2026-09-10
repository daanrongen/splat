from pathlib import Path

import pytest

from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import CaptionMetadata
from splat.domain.value_objects import CC_BY_NC_SA_4_0, MIT
from splat.handlers.diffuse import DiffuseRequest, handle
from tests.image_helpers import sample_png_bytes, write_sample_png


class FakeDiffusionBackend:
    name = "fake-diffuser"

    def __init__(self, license) -> None:
        self.license = license
        self.last_call: dict | None = None

    def diffuse(self, prompt, *, output_path: Path, **params) -> Path:
        self.last_call = {"prompt": prompt, **params}
        output_path.write_bytes(sample_png_bytes())
        return output_path


def test_handle_returns_asset_with_no_warning_for_commercial_model(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend",
        return_value=FakeDiffusionBackend(MIT),
    )

    result = handle(DiffuseRequest(prompt="a fox", model="fake-diffuser"))

    assert result.license_warning is None
    assert result.asset.content_path.read_bytes() == sample_png_bytes()
    assert result.asset.created_by == "diffuse:fake-diffuser"
    assert result.asset.metadata.output_width == 2
    assert result.asset.metadata.output_height == 2
    assert result.asset.content_size == len(sample_png_bytes())
    assert result.asset.content_sha256
    assert result.asset.created_at


def test_handle_warns_for_non_commercial_model(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend",
        return_value=FakeDiffusionBackend(CC_BY_NC_SA_4_0),
    )

    result = handle(DiffuseRequest(prompt="a fox", model="fake-diffuser"))

    assert result.license_warning is not None
    assert "fake-diffuser" in result.license_warning


def test_handle_edits_an_input_image(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    backend = FakeDiffusionBackend(MIT)
    mocker.patch("splat.handlers.diffuse.get_diffusion_backend", return_value=backend)
    cache = FilesystemManifestRepository(tmp_path / "cache")
    source = cache.put_external(
        write_sample_png(tmp_path / "image.png", (2, 2)), kind=ManifestKind.IMAGE
    )

    result = handle(
        DiffuseRequest(prompt="add rain", inputs=[source], model="fake-diffuser", strength=0.4)
    )

    assert result.asset.parent_ids == [source.id]
    assert backend.last_call["image"] is not None
    assert backend.last_call["strength"] == 0.4


def test_handle_rejects_non_image_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch(
        "splat.handlers.diffuse.get_diffusion_backend", return_value=FakeDiffusionBackend(MIT)
    )
    cache = FilesystemManifestRepository(tmp_path / "cache")
    asset = cache.put(
        "caption",
        kind=ManifestKind.CAPTION,
        content_bytes=b"a fox",
        ext="txt",
        metadata=CaptionMetadata(text_length=5),
        parent_ids=[],
        created_by="caption:test",
    )

    with pytest.raises(SplatDomainError, match="image or sticker"):
        handle(DiffuseRequest(prompt="add rain", inputs=[asset]))
