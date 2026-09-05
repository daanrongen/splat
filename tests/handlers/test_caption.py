from pathlib import Path

from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import MIT
from splat.handlers.caption import CaptionRequest, handle
from splat.registry.wiring import get_manifest_repository
from tests.image_helpers import write_sample_png


class FakeCaptionBackend:
    name = "fake-captioner"
    license = MIT

    def caption(
        self,
        image_path: Path,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **params,
    ) -> str:
        return f"{prompt} ({max_tokens}, {temperature})"


def test_handle_captions_each_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.caption.get_caption_backend", return_value=FakeCaptionBackend())
    cache = get_manifest_repository()
    asset = cache.put_external(
        write_sample_png(tmp_path / "scene.png", (3, 2)), kind=ManifestKind.IMAGE
    )

    results = handle(
        CaptionRequest(
            inputs=[asset],
            model="fake-captioner",
            prompt="Describe",
            max_tokens=12,
            temperature=0.0,
        )
    )

    assert len(results) == 1
    assert results[0].kind == ManifestKind.CAPTION
    assert results[0].content_path.read_text(encoding="utf-8") == "Describe (12, 0.0)"
    assert results[0].parent_ids == [asset.id]
    assert results[0].metadata.model == "fake-captioner"
