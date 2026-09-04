import json
import re
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from splat.adapters.cache.filesystem import FilesystemAssetCache
from splat.cli.main import app
from splat.domain.asset import AssetKind
from splat.domain.value_objects import MIT
from tests.image_helpers import write_sample_png

runner = CliRunner()


class FakeEmbeddingBackend:
    name = "fake-embedder"
    license = MIT

    def embed_image(self, image_path: Path, **params) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def embed_text(self, text: str, **params) -> np.ndarray:
        return np.array([0.0, 1.0, float(len(text))], dtype=np.float32)


def _patch_backend(mocker):
    return mocker.patch(
        "splat.handlers.embed.get_embedding_backend", return_value=FakeEmbeddingBackend()
    )


def _sample_image(tmp_path: Path) -> Path:
    return write_sample_png(tmp_path / "scene.png", (3, 2))


def test_embed_file_path_input_ndjson_output(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)

    result = runner.invoke(app, ["embed", str(_sample_image(tmp_path)), "--model", "fake-embedder"])

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["kind"] == "embedding"
    assert line["created_by"] == "embed:fake-embedder"
    assert line["metadata"]["input_type"] == "image"
    assert line["metadata"]["dimension"] == 3


def test_embed_text_ndjson_output(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)

    result = runner.invoke(app, ["embed", "--text", "red chair", "--model", "fake-embedder"])

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["kind"] == "embedding"
    assert line["metadata"]["input_type"] == "text"
    assert line["metadata"]["text_length"] == len("red chair")


def test_embed_human_output(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)
    mocker.patch("splat.cli._pipeline_io.is_piped", return_value=False)

    result = runner.invoke(app, ["embed", "--text", "red chair", "--model", "fake-embedder"])
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)

    assert result.exit_code == 0, result.output
    assert "embedded" in plain
    assert "text 3d float32" in plain


def test_embed_output_flag_writes_npy(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)
    out_path = tmp_path / "embedding.npy"

    result = runner.invoke(
        app,
        ["embed", "--text", "red chair", "--model", "fake-embedder", "-o", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert np.load(out_path).shape == (3,)


def test_embed_asset_id_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)
    cache = FilesystemAssetCache(tmp_path / "cache")
    asset = cache.put_external(_sample_image(tmp_path), kind=AssetKind.IMAGE)

    result = runner.invoke(app, ["embed", f"@{asset.id}", "--model", "fake-embedder"])

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["parent_ids"] == [asset.id]


def test_embed_stdin_ndjson_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)
    cache = FilesystemAssetCache(tmp_path / "cache")
    asset = cache.put_external(_sample_image(tmp_path), kind=AssetKind.IMAGE)

    result = runner.invoke(
        app,
        ["embed", "-", "--model", "fake-embedder"],
        input=json.dumps({"id": asset.id}) + "\n",
    )

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["parent_ids"] == [asset.id]


def test_embed_stdin_caption_asset_as_text(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)
    cache = FilesystemAssetCache(tmp_path / "cache")
    caption = cache.put(
        "caption-input",
        kind=AssetKind.CAPTION,
        content_bytes=b"red chair",
        ext="txt",
        metadata={"text_length": len("red chair")},
        parent_ids=[],
        created_by="caption:fake-captioner",
    )

    result = runner.invoke(
        app,
        ["embed", "-", "--model", "fake-embedder"],
        input=json.dumps({"id": caption.id}) + "\n",
    )

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["metadata"]["input_type"] == "text"
    assert line["parent_ids"] == [caption.id]


def test_embed_rejects_image_and_text_together(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker)

    result = runner.invoke(app, ["embed", str(_sample_image(tmp_path)), "--text", "red chair"])

    assert result.exit_code == 1
    assert "either image input or --text" in result.output
