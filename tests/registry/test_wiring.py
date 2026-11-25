from pathlib import Path

import pytest

from splat.adapters.formats.ply import PlyReader, PlyWriter
from splat.adapters.model_sources.huggingface import HuggingFaceModelSource
from splat.adapters.model_sources.stub import StubModelSource
from splat.domain.errors import UnsupportedFormat
from splat.registry.depth import DEPTH_CATALOG
from splat.registry.upscale import UPSCALE_CATALOG
from splat.registry.wiring import (
    get_client,
    get_embedding_backend,
    get_model_source,
    get_reader,
    get_reconstruction_backend,
    get_render_backend,
    get_upscale_backend,
    get_writer,
    is_known_format,
    resolve_weights,
)


class FakeModelSource:
    def __init__(self, installed=()) -> None:
        self.installed = set(installed)
        self.pulled: list[tuple[str, tuple[str, ...]]] = []

    def local_path(self, model_id, files=()):
        return Path("/store") / model_id.replace("/", "--") if model_id in self.installed else None

    def pull(self, model_id, *, files=(), on_progress=None):
        self.pulled.append((model_id, files))
        return Path("/pulled") / model_id.replace("/", "--")


def test_get_reader_known_format():
    assert isinstance(get_reader(".ply"), PlyReader)


def test_get_writer_known_format():
    assert isinstance(get_writer(".ply"), PlyWriter)


def test_get_reader_unknown_format_raises():
    with pytest.raises(UnsupportedFormat):
        get_reader(".gltf")


def test_get_writer_unknown_format_raises():
    with pytest.raises(UnsupportedFormat):
        get_writer(".gltf")


def test_is_known_format():
    assert is_known_format(".ply")
    assert not is_known_format(".gltf")


def test_get_reconstruction_backend_unknown_model_raises():
    with pytest.raises(UnsupportedFormat, match="Unknown model"):
        get_reconstruction_backend("not-a-real-model", model_source=FakeModelSource())


def test_get_reconstruction_backend_wires_known_model():
    backend = get_reconstruction_backend(
        "mlx3d-capture", model_source=FakeModelSource(), device="cpu"
    )
    assert backend.name == "mlx3d-capture"
    assert backend.required_image_count() == (3, None)


def test_get_render_backend_unknown_model_raises():
    with pytest.raises(UnsupportedFormat, match="Unknown render backend"):
        get_render_backend("not-a-real-model")


def test_get_render_backend_wires_known_model():
    backend = get_render_backend("blender")
    assert backend.name == "blender"


def test_get_embedding_backend_unknown_model_raises():
    with pytest.raises(UnsupportedFormat, match="Unknown embedding model"):
        get_embedding_backend("not-a-real-model", model_source=FakeModelSource())


def test_get_upscale_backend_unknown_model_raises():
    with pytest.raises(UnsupportedFormat, match="Unknown upscale model"):
        get_upscale_backend("not-a-real-model", model_source=FakeModelSource())


def test_get_upscale_backend_wires_known_model():
    backend = get_upscale_backend("realesrgan-mlx", model_source=FakeModelSource())
    assert backend.name == "realesrgan-mlx"
    assert backend.supported_factors == (2, 4)


def test_resolve_weights_uses_the_installed_directory_without_pulling():
    source = FakeModelSource(installed={"apple/coreml-depth-anything-v2-small"})

    paths = resolve_weights(DEPTH_CATALOG["depth-anything-v2-coreml"], source)

    assert paths == (Path("/store/apple--coreml-depth-anything-v2-small"),)
    assert source.pulled == []


def test_resolve_weights_pulls_only_the_files_a_missing_model_needs():
    source = FakeModelSource()

    resolve_weights(DEPTH_CATALOG["depth-anything-v2-coreml"], source)

    assert source.pulled == [
        ("apple/coreml-depth-anything-v2-small", ("DepthAnythingV2SmallF16.mlpackage/**",))
    ]


def test_resolve_weights_follows_the_repo_order_of_the_catalog():
    source = FakeModelSource(installed={"mlx-community/Real-ESRGAN-x4plus"})

    paths = resolve_weights(UPSCALE_CATALOG["realesrgan-mlx"], source)

    assert paths == (
        Path("/pulled/mlx-community--Real-ESRGAN-x2plus"),
        Path("/store/mlx-community--Real-ESRGAN-x4plus"),
    )


def test_resolve_weights_of_a_model_without_weights_is_empty():
    from splat.registry.gaussian import GAUSSIAN_CATALOG

    assert resolve_weights(GAUSSIAN_CATALOG["mlx3d-capture"], FakeModelSource()) == ()


def test_get_model_source_defaults_to_huggingface(monkeypatch):
    monkeypatch.delenv("SPLAT_MODEL_SOURCE", raising=False)

    assert type(get_model_source()) is HuggingFaceModelSource


def test_get_model_source_selects_the_stub(monkeypatch):
    monkeypatch.setenv("SPLAT_MODEL_SOURCE", "stub")

    assert isinstance(get_model_source(), StubModelSource)


def test_get_model_source_unknown_name_raises(monkeypatch):
    monkeypatch.setenv("SPLAT_MODEL_SOURCE", "nope")

    with pytest.raises(UnsupportedFormat, match="Available: huggingface, stub"):
        get_model_source()


def test_get_client_is_remote_when_splat_url_is_set(monkeypatch):
    monkeypatch.setenv("SPLAT_URL", "http://splat.test")
    monkeypatch.delenv("SPLAT_LOCAL", raising=False)
    monkeypatch.setattr("splat.adapters.client.http._check_server_version", lambda client: None)

    assert type(get_client()).__name__ == "RemoteSplatClient"


def test_get_client_is_local_when_splat_local_overrides_splat_url(monkeypatch):
    monkeypatch.setenv("SPLAT_URL", "http://splat.test")
    monkeypatch.setenv("SPLAT_LOCAL", "1")

    assert type(get_client()).__name__ == "LocalSplatClient"
