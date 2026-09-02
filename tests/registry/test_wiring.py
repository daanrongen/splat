import pytest

from splat.adapters.formats.ply import PlyReader, PlyWriter
from splat.domain.errors import UnsupportedFormat
from splat.registry.wiring import (
    get_reader,
    get_reconstruction_backend,
    get_writer,
    is_known_format,
)


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
    class DummyModelSource:
        def pull(self, model_id, *, revision=None):
            raise AssertionError("should not be called for unknown model")

    with pytest.raises(UnsupportedFormat, match="Unknown model"):
        get_reconstruction_backend("not-a-real-model", model_source=DummyModelSource())


def test_get_reconstruction_backend_wires_known_model(tmp_path):
    class DummyModelSource:
        def pull(self, model_id, *, revision=None):
            return tmp_path

    backend = get_reconstruction_backend("mvsplat", model_source=DummyModelSource(), device="cpu")
    assert backend.name == "mvsplat"
    assert backend.required_image_count() == (2, None)
