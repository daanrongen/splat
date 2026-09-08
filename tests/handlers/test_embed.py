from pathlib import Path

import numpy as np
import pytest

from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import CaptionMetadata
from splat.domain.value_objects import MIT
from splat.handlers.embed import EmbedRequest, handle
from splat.registry.wiring import get_manifest_repository
from tests.image_helpers import write_sample_png


class FakeEmbeddingBackend:
    name = "fake-embedder"
    license = MIT

    def embed_image(self, image_path: Path, **params) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_text(self, text: str, **params) -> np.ndarray:
        return np.array([0.0, 1.0], dtype=np.float32)


def test_handle_embeds_each_image_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.embed.get_embedding_backend", return_value=FakeEmbeddingBackend())
    cache = get_manifest_repository()
    asset = cache.put_external(
        write_sample_png(tmp_path / "scene.png", (3, 2)), kind=ManifestKind.IMAGE
    )

    results = handle(EmbedRequest(inputs=[asset], model="fake-embedder", device="cpu"))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.EMBEDDING
    assert results[0].parent_ids == [asset.id]


def test_handle_embeds_text_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.embed.get_embedding_backend", return_value=FakeEmbeddingBackend())

    results = handle(EmbedRequest(text="red chair", model="fake-embedder", device="cpu"))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.EMBEDDING
    assert results[0].metadata.input_type == "text"


def test_handle_embeds_caption_asset_as_text(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.embed.get_embedding_backend", return_value=FakeEmbeddingBackend())
    cache = get_manifest_repository()
    caption = cache.put(
        "caption-input",
        kind=ManifestKind.CAPTION,
        content_bytes=b"red chair",
        ext="txt",
        metadata=CaptionMetadata(text_length=len("red chair")),
        parent_ids=[],
        created_by="caption:fake-captioner",
    )

    results = handle(EmbedRequest(inputs=[caption], model="fake-embedder", device="cpu"))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.EMBEDDING
    assert results[0].metadata.input_type == "text"
    assert results[0].parent_ids == [caption.id]


def test_handle_rejects_invalid_mode_combinations(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.embed.get_embedding_backend", return_value=FakeEmbeddingBackend())
    cache = get_manifest_repository()
    asset = cache.put_external(
        write_sample_png(tmp_path / "scene.png", (3, 2)), kind=ManifestKind.IMAGE
    )

    with pytest.raises(Exception, match="either image input or --text"):
        handle(EmbedRequest(inputs=[asset], text="red chair"))

    with pytest.raises(Exception, match="either image input or --text"):
        handle(EmbedRequest())
