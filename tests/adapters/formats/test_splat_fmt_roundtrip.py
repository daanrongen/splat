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
