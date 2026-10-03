import numpy as np
import pytest

from splat.adapters.cache.filesystem import FilesystemManifestRepository
from splat.adapters.formats.image import encode_png
from splat.domain.errors import ContractViolation
from splat.domain.gaussians import GaussianCloudMetadata
from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import (
    CaptionMetadata,
    DepthMetadata,
    EmbeddingMetadata,
    MeshMetadata,
    RasterMetadata,
    StickerMetadata,
    check_contract,
    metadata_from_dict,
)
from tests.image_helpers import depth_npy, sample_rgb, sample_rgba, tiny_png

STICKER = StickerMetadata(bbox=(0, 0, 2, 2), score=1.0, area=4)
PLY = b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nend_header\n"


@pytest.mark.parametrize(
    ("kind", "metadata", "content"),
    [
        (
            ManifestKind.IMAGE,
            RasterMetadata(output_width=2, output_height=2),
            encode_png(sample_rgb()),
        ),
        (ManifestKind.STICKER, STICKER, encode_png(sample_rgba())),
        (ManifestKind.CAPTION, CaptionMetadata(text_length=5), b"a fox"),
        (ManifestKind.DEPTH_MAP, DepthMetadata(width=2, height=2), depth_npy()),
        (ManifestKind.GAUSSIAN_CLOUD, GaussianCloudMetadata(point_count=3), PLY),
        (ManifestKind.SHAPE_3D, MeshMetadata(vertex_count=3, face_count=1), b"glb"),
    ],
)
def test_canonical_outputs_pass(kind, metadata, content):
    check_contract(kind, metadata, content)


@pytest.mark.parametrize(
    ("kind", "metadata", "content"),
    [
        (ManifestKind.IMAGE, RasterMetadata(), b"not a png"),
        (ManifestKind.IMAGE, RasterMetadata(output_width=4, output_height=4), tiny_png("x")),
        (ManifestKind.STICKER, STICKER, encode_png(sample_rgb())),
        (ManifestKind.CAPTION, CaptionMetadata(text_length=0), b"  "),
        (ManifestKind.DEPTH_MAP, DepthMetadata(), b""),
        (ManifestKind.DEPTH_MAP, DepthMetadata(units="feet"), depth_npy()),
        (
            ManifestKind.GAUSSIAN_CLOUD,
            GaussianCloudMetadata(coordinate_convention="colmap", point_count=3),
            PLY,
        ),
        (ManifestKind.GAUSSIAN_CLOUD, GaussianCloudMetadata(point_count=9), PLY),
        (ManifestKind.SHAPE_3D, MeshMetadata(), b"glb"),
        (ManifestKind.IMAGE, CaptionMetadata(text_length=1), tiny_png("x")),
    ],
)
def test_violations_are_rejected(kind, metadata, content):
    with pytest.raises(ContractViolation):
        check_contract(kind, metadata, content)


def test_embedding_shape_must_match_the_array():
    from io import BytesIO

    buf = BytesIO()
    np.save(buf, np.zeros(4, dtype=np.float32))
    meta = EmbeddingMetadata(
        input_type="text", dtype="float32", shape=[4], dimension=4, normalized=True, model="m"
    )
    check_contract(ManifestKind.EMBEDDING, meta, buf.getvalue())
    meta.shape = [8]
    with pytest.raises(ContractViolation):
        check_contract(ManifestKind.EMBEDDING, meta, buf.getvalue())


def test_put_rejects_a_stage_output_but_not_an_external_import(tmp_path):
    cache = FilesystemManifestRepository(tmp_path)
    args = {"kind": ManifestKind.IMAGE, "content_bytes": b"junk", "ext": "png"}
    args |= {"metadata": RasterMetadata(), "parent_ids": []}

    with pytest.raises(ContractViolation):
        cache.put("staged", created_by="diffuse:test", **args)
    assert cache.put("imported", created_by="external", **args).id == "imported"


def test_untyped_mesh_metadata_still_loads():
    meta = metadata_from_dict(
        ManifestKind.SHAPE_3D, {"extra": {"vertex_count": 3, "face_count": 1}}
    )
    assert (meta.vertex_count, meta.face_count) == (3, 1)
