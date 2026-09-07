from pathlib import Path

import pytest

import splat
from splat.domain.manifest import Manifest, ManifestKind
from splat.domain.manifest_metadata import RasterMetadata
from tests.image_helpers import write_sample_png


def _manifest(path=Path("/tmp/whatever.png"), kind=ManifestKind.IMAGE) -> Manifest:
    return Manifest(
        id="abc123",
        kind=kind,
        content_path=path,
        metadata=RasterMetadata(output_width=1, output_height=1),
    )


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def diffuse(self, request):
        self.calls.append(("diffuse", request))
        return "diffuse-result"

    def caption(self, request):
        self.calls.append(("caption", request))
        return ["caption-result"]

    def segment(self, request):
        self.calls.append(("segment", request))
        return ["segment-result"]

    def depth(self, request):
        self.calls.append(("depth", request))
        return ["depth-result"]

    def upscale(self, request):
        self.calls.append(("upscale", request))
        return ["upscale-result"]

    def embed(self, request):
        self.calls.append(("embed", request))
        return ["embed-result"]

    def gaussian(self, request):
        self.calls.append(("gaussian", request))
        return ["gaussian-result"]

    def tools_convert(self, request):
        self.calls.append(("tools_convert", request))
        return "convert-result"

    def tools_compress(self, request):
        self.calls.append(("tools_compress", request))
        return "compress-result"

    def info(self, path):
        self.calls.append(("info", path))
        return "info-result"

    def validate(self, path, *, strict=False):
        self.calls.append(("validate", (path, strict)))
        return "validate-result"


@pytest.fixture
def fake_client(mocker):
    client = _FakeClient()
    mocker.patch("splat.api.get_client", return_value=client)
    return client


def test_diffuse_builds_request_and_delegates(fake_client):
    result = splat.diffuse("a fox", steps=4)

    assert result == "diffuse-result"
    name, request = fake_client.calls[0]
    assert name == "diffuse"
    assert request.prompt == "a fox"
    assert request.steps == 4


def test_caption_resolves_path_input(fake_client, tmp_path):
    image_path = write_sample_png(tmp_path / "a.png", (2, 2))

    result = splat.caption(image_path)

    assert result == ["caption-result"]
    _, request = fake_client.calls[0]
    assert len(request.inputs) == 1
    assert request.inputs[0].kind == ManifestKind.IMAGE
    assert request.inputs[0].content_path.read_bytes() == image_path.read_bytes()


def test_caption_accepts_list_of_inputs(fake_client, tmp_path):
    a = write_sample_png(tmp_path / "a.png", (2, 2))
    b = write_sample_png(tmp_path / "b.png", (2, 2))

    splat.caption([a, b])

    _, request = fake_client.calls[0]
    assert len(request.inputs) == 2


def test_caption_passes_through_manifest_unchanged(fake_client):
    manifest = _manifest()

    splat.caption(manifest)

    _, request = fake_client.calls[0]
    assert request.inputs == [manifest]


def test_caption_resolves_at_id_via_manifest_load(fake_client, mocker):
    manifest = _manifest()
    mocker.patch("splat.domain.manifest.Manifest.load", return_value=manifest)

    splat.caption("@abc123")

    _, request = fake_client.calls[0]
    assert request.inputs == [manifest]


def test_segment_builds_request(fake_client):
    manifest = _manifest()

    result = splat.segment(manifest, max_stickers=5)

    assert result == ["segment-result"]
    _, request = fake_client.calls[0]
    assert request.max_stickers == 5
    assert request.inputs == [manifest]


def test_embed_with_text_only_skips_input_resolution(fake_client):
    splat.embed(text="a red chair")

    _, request = fake_client.calls[0]
    assert request.inputs is None
    assert request.text == "a red chair"


def test_gaussian_builds_request(fake_client):
    manifest = _manifest(kind=ManifestKind.IMAGE)

    result = splat.gaussian(manifest, quality="balanced")

    assert result == ["gaussian-result"]
    _, request = fake_client.calls[0]
    assert request.quality == "balanced"
    assert request.inputs == [manifest]


def test_render_bypasses_client_and_calls_handler_directly(mocker, fake_client):
    handle_render = mocker.patch("splat.handlers.render.handle", return_value=["render-result"])
    manifest = _manifest(kind=ManifestKind.GAUSSIAN_CLOUD)

    result = splat.render(manifest, engine="eevee")

    assert result == ["render-result"]
    handle_render.assert_called_once()
    request = handle_render.call_args[0][0]
    assert request.engine == "eevee"
    assert request.inputs == [manifest]
    assert not fake_client.calls  # render never goes through get_client()


def test_tools_compress_builds_request(fake_client, tmp_path):
    result = splat.tools_compress(tmp_path / "in.ply", tmp_path / "out.ply", profile="archival")

    assert result == "compress-result"
    _, request = fake_client.calls[0]
    assert request.profile == "archival"
    assert request.input_path == tmp_path / "in.ply"


def test_tools_convert_builds_request(fake_client, tmp_path):
    result = splat.tools_convert(tmp_path / "in.ply", tmp_path / "out.splat")

    assert result == "convert-result"
    _, request = fake_client.calls[0]
    assert request.output_path == tmp_path / "out.splat"


def test_info_and_validate_delegate(fake_client, tmp_path):
    assert splat.info(tmp_path / "a.ply") == "info-result"
    assert splat.validate(tmp_path / "a.ply", strict=True) == "validate-result"
    assert fake_client.calls[0] == ("info", tmp_path / "a.ply")
    assert fake_client.calls[1] == ("validate", (tmp_path / "a.ply", True))
