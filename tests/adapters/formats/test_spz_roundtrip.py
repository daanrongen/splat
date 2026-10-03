import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from splat.adapters.formats.ply import PlyWriter
from splat.adapters.formats.spz import SpzReader, SpzWriter
from splat.domain.errors import SplatDomainError
from splat.domain.gaussians import to_convention

FIXTURE = Path(__file__).parents[2] / "fixtures" / "niantic-v3.spz"


def test_round_trip_stays_within_spz_quantization(tmp_path, synthetic_cloud):
    path = tmp_path / "scene.spz"
    SpzWriter().write(synthetic_cloud, path)
    loaded = SpzReader().read(path)

    assert loaded.point_count == synthetic_cloud.point_count
    assert loaded.sh_degree == synthetic_cloud.sh_degree
    np.testing.assert_allclose(loaded.means, synthetic_cloud.means, atol=2.0 / 4096)
    np.testing.assert_allclose(loaded.to_log_scales(), synthetic_cloud.to_log_scales(), atol=1 / 16)
    dots = np.abs((loaded.rotations * synthetic_cloud.rotations).sum(axis=1))
    assert dots.min() > 0.999
    np.testing.assert_allclose(loaded.sh_rest, synthetic_cloud.sh_rest, atol=0.07)


def test_writes_colmap_clouds_in_the_canonical_frame(tmp_path, synthetic_cloud):
    synthetic_cloud.metadata.coordinate_convention = "colmap"
    path = tmp_path / "scene.spz"
    SpzWriter().write(synthetic_cloud, path)

    expected = to_convention(synthetic_cloud, "opengl").means
    np.testing.assert_allclose(SpzReader().read(path).means, expected, atol=2.0 / 4096)


def test_is_much_smaller_than_ply_and_keeps_sh(tmp_path, synthetic_cloud):
    ply, spz = tmp_path / "scene.ply", tmp_path / "scene.spz"
    PlyWriter().write(synthetic_cloud, ply)
    SpzWriter().write(synthetic_cloud, spz)

    assert spz.stat().st_size < ply.stat().st_size / 3
    assert SpzReader().read(spz).sh_rest is not None


def test_reads_a_file_written_by_niantics_reference_library():
    """Generated with nianticlabs/spz v3.0.0 (PackOptions version=3, RUB) from this cloud."""
    rng = np.random.default_rng(7)
    n = 20
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    positions = rng.uniform(-5, 5, n * 3).reshape(n, 3)
    scales = rng.uniform(-6, 1, n * 3).reshape(n, 3)
    alphas = rng.uniform(-4, 4, n)
    colors = rng.uniform(-1.5, 1.5, n * 3).reshape(n, 3)
    sh = rng.uniform(-0.5, 0.5, n * 9).reshape(n, 3, 3)

    cloud = SpzReader().read(FIXTURE)

    np.testing.assert_allclose(cloud.means, positions, atol=1 / 4096)
    np.testing.assert_allclose(cloud.scales, scales, atol=1 / 16)
    assert np.abs((cloud.rotations * quats[:, [3, 0, 1, 2]]).sum(axis=1)).min() > 0.999
    np.testing.assert_allclose(cloud.opacities, 1 / (1 + np.exp(-alphas)), atol=1 / 255)
    np.testing.assert_allclose(cloud.sh_dc, colors, atol=0.014)
    np.testing.assert_allclose(cloud.sh_rest, sh, atol=0.07)


def test_reads_v2_first_three_rotations(tmp_path):
    header = struct.pack("<IIIBBBB", 0x5053474E, 2, 1, 0, 12, 0, 0)
    body = bytes(9) + bytes([255]) + bytes([128] * 3) + bytes([160] * 3) + bytes([128] * 3)
    path = tmp_path / "v2.spz"
    path.write_bytes(gzip.compress(header + body))

    cloud = SpzReader().read(path)

    np.testing.assert_allclose(cloud.rotations[0], [1, 0, 0, 0], atol=0.01)
    np.testing.assert_allclose(cloud.scales[0], [0, 0, 0])


def test_rejects_files_it_cannot_read(tmp_path):
    path = tmp_path / "bad.spz"
    path.write_bytes(b"NGSP not gzip")

    with pytest.raises(SplatDomainError, match="not a gzip"):
        SpzReader().read(path)
