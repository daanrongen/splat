from pathlib import Path

from splat.application.pipeline import run_caption
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import MIT
from splat.registry.wiring import get_manifest_repository
from tests.image_helpers import write_sample_png


class CountingCaptionBackend:
    name = "fake-captioner"
    license = MIT

    def __init__(self) -> None:
        self.calls = 0

    def caption(
        self,
        image_path: Path,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **params,
    ) -> str:
        self.calls += 1
        return f"{prompt} caption {max_tokens} {temperature}"


def _sample_asset(tmp_path: Path):
    cache = get_manifest_repository()
    image_path = write_sample_png(tmp_path / "scene.png", (3, 2))
    return cache.put_external(image_path, kind=ManifestKind.IMAGE)


def test_run_caption_creates_text_asset(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    asset = _sample_asset(tmp_path)
    backend = CountingCaptionBackend()

    result = run_caption(
        backend,
        get_manifest_repository(),
        model_name="fake-captioner",
        input_asset=asset,
        params={"prompt": "Describe", "max_tokens": 12, "temperature": 0.0},
    )

    assert result.kind == ManifestKind.CAPTION
    assert result.content_path.suffix == ".txt"
    assert result.content_path.read_text(encoding="utf-8") == "Describe caption 12 0.0"
    assert result.metadata.text_length == len("Describe caption 12 0.0")
    assert result.parent_ids == [asset.id]
    assert result.created_by == "caption:fake-captioner"


def test_run_caption_reuses_cache_for_same_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    asset = _sample_asset(tmp_path)
    backend = CountingCaptionBackend()
    params = {"prompt": "Describe", "max_tokens": 12, "temperature": 0.0}

    first = run_caption(
        backend,
        get_manifest_repository(),
        model_name="fake-captioner",
        input_asset=asset,
        params=params,
    )
    second = run_caption(
        backend,
        get_manifest_repository(),
        model_name="fake-captioner",
        input_asset=asset,
        params=params,
    )

    assert first.id == second.id
    assert backend.calls == 1


def test_run_caption_cache_key_includes_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    asset = _sample_asset(tmp_path)
    backend = CountingCaptionBackend()

    first = run_caption(
        backend,
        get_manifest_repository(),
        model_name="fake-captioner",
        input_asset=asset,
        params={"prompt": "Describe", "max_tokens": 12, "temperature": 0.0},
    )
    second = run_caption(
        backend,
        get_manifest_repository(),
        model_name="fake-captioner",
        input_asset=asset,
        params={"prompt": "Summarize", "max_tokens": 12, "temperature": 0.0},
    )

    assert first.id != second.id
    assert backend.calls == 2
