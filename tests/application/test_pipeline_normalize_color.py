from pathlib import Path

from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.adapters.formats.image import write_png
from splat.application.pipeline import run_normalize_color
from splat.domain.manifest import ManifestKind
from tests.image_helpers import sample_rgb


def _image_asset(cache: FilesystemManifestRepository, tmp_path: Path, name: str, fill: int):
    path = tmp_path / name
    write_png(path, sample_rgb((2, 2), color=(fill, fill, fill)))
    return cache.put_external(path, kind=ManifestKind.IMAGE)


def test_run_normalize_color_creates_one_output_per_input(tmp_path):
    cache = FilesystemManifestRepository(tmp_path / "cache")
    a = _image_asset(cache, tmp_path, "a.png", fill=50)
    b = _image_asset(cache, tmp_path, "b.png", fill=150)

    results = run_normalize_color(cache, input_assets=[a, b], params={})

    assert len(results) == 2
    assert all(r.kind == ManifestKind.IMAGE for r in results)
    assert results[0].parent_ids == [a.id]
    assert results[1].parent_ids == [b.id]
    assert results[0].created_by == "tools:normalize.color"


def test_run_normalize_color_reuses_cache_for_same_cohort(tmp_path, mocker):
    cache = FilesystemManifestRepository(tmp_path / "cache")
    a = _image_asset(cache, tmp_path, "a.png", fill=50)
    b = _image_asset(cache, tmp_path, "b.png", fill=150)
    spy = mocker.spy(cache, "put")

    first = run_normalize_color(cache, input_assets=[a, b], params={})
    second = run_normalize_color(cache, input_assets=[a, b], params={})

    assert [r.id for r in first] == [r.id for r in second]
    assert spy.call_count == 2  # only the first call wrote anything


def test_run_normalize_color_invalidates_cache_when_cohort_changes(tmp_path):
    cache = FilesystemManifestRepository(tmp_path / "cache")
    a = _image_asset(cache, tmp_path, "a.png", fill=50)
    b = _image_asset(cache, tmp_path, "b.png", fill=150)
    c = _image_asset(cache, tmp_path, "c.png", fill=100)

    pair_result = run_normalize_color(cache, input_assets=[a, b], params={})
    triple_result = run_normalize_color(cache, input_assets=[a, b, c], params={})

    assert pair_result[0].id != triple_result[0].id
