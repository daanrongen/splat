from dataclasses import replace
from pathlib import Path

import numpy as np

from splat.adapters.formats.splat_fmt import SplatFormatReader, SplatFormatWriter


def test_splat_roundtrip_preserves_point_count_and_is_lossy_on_sh(tmp_path: Path, synthetic_cloud):
    out = tmp_path / "scene.splat"
    writer = SplatFormatWriter()
    warnings = writer.supports(synthetic_cloud)
    assert any("SH degree" in w for w in warnings)

    writer.write(synthetic_cloud, out)
    loaded = SplatFormatReader().read(out)

    assert loaded.point_count == synthetic_cloud.point_count
    assert loaded.sh_degree == 0
    assert loaded.sh_rest is None

    # positions are float32, losslessly round-tripped
    np.testing.assert_allclose(loaded.means, synthetic_cloud.means, atol=1e-4)

    # color/opacity/rotation are quantized to 8 bits -> coarse tolerance
    np.testing.assert_allclose(
        loaded.to_activated_opacities(),
        synthetic_cloud.to_activated_opacities(),
        atol=1 / 255 + 1e-3,
    )


def test_splat_file_size_matches_32_bytes_per_point(tmp_path: Path, synthetic_cloud):
    out = tmp_path / "scene.splat"
    SplatFormatWriter().write(synthetic_cloud, out)
    assert out.stat().st_size == synthetic_cloud.point_count * 32


def test_splat_write_converts_colmap_coordinates_instead_of_relabelling(
    tmp_path: Path, synthetic_cloud
):
    """A colmap-convention cloud written to .splat must actually land in
    opengl coordinates - the reader always reports "opengl" (the format has
    no field to store the convention in), so silently relabelling without
    transforming would make every downstream consumer read it upside down."""
    colmap_cloud = replace(
        synthetic_cloud, metadata=replace(synthetic_cloud.metadata, coordinate_convention="colmap")
    )
    out = tmp_path / "scene.splat"
    SplatFormatWriter().write(colmap_cloud, out)
    loaded = SplatFormatReader().read(out)

    expected = colmap_cloud.means @ np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    np.testing.assert_allclose(loaded.means, expected, atol=1e-4)
    assert not np.allclose(loaded.means, colmap_cloud.means, atol=1e-4)
