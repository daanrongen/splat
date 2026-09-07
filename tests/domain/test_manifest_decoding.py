import numpy as np

from splat.adapters.formats.ply import PlyWriter
from splat.domain.manifest import Manifest, ManifestKind
from splat.domain.manifest_metadata import CaptionMetadata, EmbeddingMetadata, RasterMetadata
from tests.image_helpers import write_sample_png


def _manifest(content_path, kind, metadata) -> Manifest:
    return Manifest(id="x", kind=kind, content_path=content_path, metadata=metadata)


def test_as_image_decodes_png(tmp_path):
    path = write_sample_png(tmp_path / "a.png", (4, 4))
    manifest = _manifest(path, ManifestKind.IMAGE, RasterMetadata(output_width=4, output_height=4))

    image = manifest.as_image()

    assert image.shape[:2] == (4, 4)


def test_as_text_reads_utf8(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("a red fox", encoding="utf-8")
    manifest = _manifest(path, ManifestKind.CAPTION, CaptionMetadata(text_length=9, model="m"))

    assert manifest.as_text() == "a red fox"


def test_as_array_loads_npy(tmp_path):
    path = tmp_path / "a.npy"
    np.save(path, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    manifest = _manifest(
        path,
        ManifestKind.EMBEDDING,
        EmbeddingMetadata(
            input_type="text", dtype="float32", shape=[3], dimension=3, normalized=True, model="m"
        ),
    )

    np.testing.assert_allclose(manifest.as_array(), [1.0, 2.0, 3.0])


def test_as_gaussian_cloud_reads_ply(tmp_path, synthetic_cloud):
    path = tmp_path / "scene.ply"
    PlyWriter().write(synthetic_cloud, path)
    manifest = _manifest(
        path, ManifestKind.GAUSSIAN_CLOUD, RasterMetadata(output_width=0, output_height=0)
    )

    cloud = manifest.as_gaussian_cloud()

    assert cloud.point_count == synthetic_cloud.point_count


def test_load_fetches_from_manifest_repository(mocker, tmp_path):
    fake_manifest = _manifest(
        tmp_path / "a.png", ManifestKind.IMAGE, RasterMetadata(output_width=1, output_height=1)
    )
    fake_cache = mocker.Mock()
    fake_cache.get.return_value = fake_manifest
    mocker.patch("splat.registry.wiring.get_manifest_repository", return_value=fake_cache)

    result = Manifest.load("@abc123")

    fake_cache.get.assert_called_once_with("abc123")
    assert result is fake_manifest


def test_load_accepts_id_without_at_prefix(mocker, tmp_path):
    fake_manifest = _manifest(
        tmp_path / "a.png", ManifestKind.IMAGE, RasterMetadata(output_width=1, output_height=1)
    )
    fake_cache = mocker.Mock()
    fake_cache.get.return_value = fake_manifest
    mocker.patch("splat.registry.wiring.get_manifest_repository", return_value=fake_cache)

    Manifest.load("abc123")

    fake_cache.get.assert_called_once_with("abc123")
