from pathlib import Path

import pytest

from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.adapters.formats.image import write_png
from splat.domain.errors import WrongManifestKind
from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import CaptionMetadata
from splat.handlers.tools.normalize_color import NormalizeColorRequest, handle
from tests.image_helpers import sample_rgb


def _image_asset(cache: FilesystemManifestRepository, tmp_path: Path, name: str, fill: int):
    path = tmp_path / name
    write_png(path, sample_rgb((2, 2), color=(fill, fill, fill)))
    return cache.put_external(path, kind=ManifestKind.IMAGE)


def test_handle_corrects_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemManifestRepository(tmp_path / "cache")
    a = _image_asset(cache, tmp_path, "a.png", fill=50)
    b = _image_asset(cache, tmp_path, "b.png", fill=150)

    results = handle(NormalizeColorRequest(inputs=[a, b]))

    assert len(results) == 2
    assert all(r.content_path.exists() for r in results)


def test_handle_rejects_non_colorlike_input(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemManifestRepository(tmp_path / "cache")
    caption_asset = cache.put(
        "captionkey1",
        kind=ManifestKind.CAPTION,
        content_bytes=b"a caption",
        ext="txt",
        metadata=CaptionMetadata(text_length=9, model="m"),
        parent_ids=[],
        created_by="caption:m",
    )

    with pytest.raises(WrongManifestKind):
        handle(NormalizeColorRequest(inputs=[caption_asset]))
