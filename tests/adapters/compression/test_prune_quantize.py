import pytest

from splat.adapters.compression.prune_quantize import PruneQuantizeCompressor


def test_blue_noise_pruning_reduces_to_target_count(synthetic_cloud):
    result = PruneQuantizeCompressor().compress(
        synthetic_cloud, profile="archival", pruning="blue-noise", target_count=10
    )
    assert result.point_count == 10


def test_blue_noise_pruning_requires_target_count(synthetic_cloud):
    with pytest.raises(ValueError, match="target_count"):
        PruneQuantizeCompressor().compress(synthetic_cloud, pruning="blue-noise")


def test_unknown_pruning_strategy_raises(synthetic_cloud):
    with pytest.raises(ValueError, match="Unknown pruning strategy"):
        PruneQuantizeCompressor().compress(synthetic_cloud, pruning="not-a-real-strategy")


def test_blue_noise_pruning_composes_with_threshold_prefilter(synthetic_cloud):
    # web-delivery's opacity/outlier thresholds run first; blue-noise then reduces
    # whatever survives that to the target count.
    result = PruneQuantizeCompressor().compress(
        synthetic_cloud, profile="web-delivery", pruning="blue-noise", target_count=5
    )
    assert result.point_count == 5
