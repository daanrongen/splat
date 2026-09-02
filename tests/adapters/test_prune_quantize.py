from splat.adapters.compression.prune_quantize import PruneQuantizeCompressor


def test_web_delivery_drops_sh_rest_and_prunes(synthetic_cloud):
    result = PruneQuantizeCompressor().compress(synthetic_cloud, profile="web-delivery")
    assert result.sh_degree == 0
    assert result.sh_rest is None
    assert result.point_count <= synthetic_cloud.point_count


def test_archival_preserves_sh_and_all_points(synthetic_cloud):
    result = PruneQuantizeCompressor().compress(synthetic_cloud, profile="archival")
    assert result.sh_degree == synthetic_cloud.sh_degree
    assert result.point_count == synthetic_cloud.point_count


def test_unknown_profile_raises(synthetic_cloud):
    import pytest

    with pytest.raises(ValueError, match="Unknown compression profile"):
        PruneQuantizeCompressor().compress(synthetic_cloud, profile="nonsense")
