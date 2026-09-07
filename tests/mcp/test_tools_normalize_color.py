from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.adapters.formats.image import write_png
from splat.domain.manifest import ManifestKind
from tests.image_helpers import sample_rgb


def _image_asset(cache: FilesystemManifestRepository, tmp_path, name: str, fill: int):
    path = tmp_path / name
    write_png(path, sample_rgb((2, 2), color=(fill, fill, fill)))
    return cache.put_external(path, kind=ManifestKind.IMAGE)


def test_normalize_color_corrects_images(tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemManifestRepository(tmp_path / "cache")
    a = _image_asset(cache, tmp_path, "a.png", fill=50)
    b = _image_asset(cache, tmp_path, "b.png", fill=150)

    result = call_tool("tools_normalize_color", image_asset_ids=[a.id, b.id])

    assert result.is_error is False
