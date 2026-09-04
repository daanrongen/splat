from pathlib import Path

import numpy as np

from splat.application.pipeline import run_embed_image, run_embed_text
from splat.domain.asset import AssetKind
from splat.domain.value_objects import MIT
from splat.registry.wiring import get_asset_cache
from tests.image_helpers import write_sample_png


class CountingEmbeddingBackend:
    name = "fake-embedder"
    license = MIT

    def __init__(self) -> None:
        self.image_calls = 0
        self.text_calls = 0

    def embed_image(self, image_path: Path, **params) -> np.ndarray:
        self.image_calls += 1
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def embed_text(self, text: str, **params) -> np.ndarray:
        self.text_calls += 1
        return np.array([0.0, 1.0, len(text)], dtype=np.float64)


def _sample_asset(tmp_path: Path):
    cache = get_asset_cache()
    image_path = write_sample_png(tmp_path / "scene.png", (3, 2))
    return cache.put_external(image_path, kind=AssetKind.IMAGE)


def test_run_embed_image_creates_embedding_asset(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    asset = _sample_asset(tmp_path)
    backend = CountingEmbeddingBackend()

    result = run_embed_image(
        backend,
        get_asset_cache(),
        model_name="fake-embedder",
        input_asset=asset,
        params={"device": "cpu"},
    )

    vector = np.load(result.content_path)
    assert result.kind == AssetKind.EMBEDDING
    assert result.content_path.suffix == ".npy"
    assert vector.dtype == np.float32
    assert vector.shape == (3,)
    assert result.metadata["input_type"] == "image"
    assert result.metadata["dtype"] == "float32"
    assert result.metadata["shape"] == [3]
    assert result.metadata["dimension"] == 3
    assert result.metadata["normalized"] is True
    assert result.parent_ids == [asset.id]
    assert result.created_by == "embed:fake-embedder"


def test_run_embed_image_reuses_cache_for_same_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    asset = _sample_asset(tmp_path)
    backend = CountingEmbeddingBackend()
    params = {"device": "cpu"}

    first = run_embed_image(
        backend, get_asset_cache(), model_name="fake-embedder", input_asset=asset, params=params
    )
    second = run_embed_image(
        backend, get_asset_cache(), model_name="fake-embedder", input_asset=asset, params=params
    )

    assert first.id == second.id
    assert backend.image_calls == 1


def test_run_embed_text_hashes_text_and_reuses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    backend = CountingEmbeddingBackend()
    params = {"device": "cpu"}

    first = run_embed_text(
        backend, get_asset_cache(), model_name="fake-embedder", text="red chair", params=params
    )
    second = run_embed_text(
        backend, get_asset_cache(), model_name="fake-embedder", text="red chair", params=params
    )

    assert first.id == second.id
    assert backend.text_calls == 1
    assert first.metadata["input_type"] == "text"
    assert first.metadata["text_length"] == len("red chair")
    assert "red chair" not in first.metadata.values()
    assert first.parent_ids == []


def test_run_embed_text_cache_key_includes_text(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    backend = CountingEmbeddingBackend()

    first = run_embed_text(
        backend, get_asset_cache(), model_name="fake-embedder", text="red chair", params={}
    )
    second = run_embed_text(
        backend, get_asset_cache(), model_name="fake-embedder", text="blue chair", params={}
    )

    assert first.id != second.id
    assert backend.text_calls == 2
