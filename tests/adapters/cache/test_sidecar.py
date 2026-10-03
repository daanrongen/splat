import json

from tests.image_helpers import tiny_png

from splat.adapters.cache.filesystem import FilesystemManifestRepository, sidecar_path
from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import RasterMetadata


def _produce(cache, manifest_id, content, parent_ids=()):
    return cache.put(
        manifest_id,
        kind=ManifestKind.IMAGE,
        content_bytes=content,
        ext="png",
        metadata=RasterMetadata(output_width=1, output_height=1),
        params={"prompt": manifest_id},
        parent_ids=list(parent_ids),
        created_by="diffuse:test",
    )


def test_sidecar_restores_identity_in_a_fresh_cache(tmp_path):
    source = FilesystemManifestRepository(tmp_path / "a")
    asset = _produce(source, "robot", tiny_png("robot"), parent_ids=["prompt-id"])
    out = tmp_path / "robot.png"
    out.write_bytes(tiny_png("robot"))
    source.write_sidecar(asset.id, out)

    restored = FilesystemManifestRepository(tmp_path / "b").put_external(
        out, kind=ManifestKind.IMAGE
    )

    assert restored.id == "robot"
    assert restored.created_by == "diffuse:test"
    assert restored.params == {"prompt": "robot"}
    assert restored.parent_ids == ["prompt-id"]
    assert restored.metadata.output_width == 1


def test_derived_file_points_back_at_its_source(tmp_path):
    cache = FilesystemManifestRepository(tmp_path / "cache")
    asset = _produce(cache, "robot", tiny_png("robot"))
    converted = tmp_path / "robot.splat"
    converted.write_bytes(b"converted-bytes")
    cache.write_sidecar(asset.id, converted)

    imported = cache.put_external(converted, kind=ManifestKind.GAUSSIAN_CLOUD)

    assert imported.created_by == "external"
    assert imported.parent_ids == ["robot"]


def test_malformed_sidecar_is_ignored(tmp_path):
    cache = FilesystemManifestRepository(tmp_path / "cache")
    image = tmp_path / "photo.png"
    image.write_bytes(b"photo-bytes")
    sidecar_path(image).write_text("{not json")

    imported = cache.put_external(image, kind=ManifestKind.GAUSSIAN_CLOUD)

    assert imported.created_by == "external"
    assert imported.parent_ids == []


def test_export_round_trips_the_whole_lineage(tmp_path):
    source = FilesystemManifestRepository(tmp_path / "a")
    _produce(source, "base", tiny_png("base"))
    _produce(source, "upscaled", tiny_png("upscaled"), parent_ids=["base"])
    out_dir = tmp_path / "export"

    written = source.export("upscaled", out_dir)

    assert {p.name for p in written} == {"upscaled.png", "base.png"}
    assert json.loads(sidecar_path(out_dir / "upscaled.png").read_text())["schema"] == 1
    fresh = FilesystemManifestRepository(tmp_path / "b")
    fresh.put_external(out_dir / "upscaled.png", kind=ManifestKind.IMAGE)
    assert {m.id for m in fresh.list()} == {"upscaled", "base"}
