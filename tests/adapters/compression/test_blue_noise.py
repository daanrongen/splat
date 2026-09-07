import numpy as np

from splat.adapters.compression.blue_noise import weighted_sample_elimination


def test_target_count_at_or_above_input_keeps_everything():
    points = np.random.default_rng(0).uniform(-1, 1, size=(10, 3)).astype(np.float32)
    importance = np.ones(10, dtype=np.float32)
    assert weighted_sample_elimination(points, importance, 10).all()
    assert weighted_sample_elimination(points, importance, 20).all()


def test_target_count_zero_or_negative_keeps_nothing():
    points = np.random.default_rng(0).uniform(-1, 1, size=(5, 3)).astype(np.float32)
    importance = np.ones(5, dtype=np.float32)
    assert not weighted_sample_elimination(points, importance, 0).any()
    assert not weighted_sample_elimination(points, importance, -3).any()


def test_reduces_to_exact_target_count():
    rng = np.random.default_rng(1)
    points = rng.uniform(-1, 1, size=(200, 3)).astype(np.float32)
    importance = rng.uniform(0, 1, size=200).astype(np.float32)
    keep = weighted_sample_elimination(points, importance, 50)
    assert keep.sum() == 50


def test_prefers_thinning_a_dense_cluster_over_dropping_an_isolated_point():
    rng = np.random.default_rng(2)
    cluster = rng.normal(scale=0.01, size=(19, 3)).astype(np.float32)
    isolated = np.array(
        [[100.0, 100.0, 100.0]], dtype=np.float32
    )  # far outside any elimination radius
    points = np.concatenate([cluster, isolated], axis=0)
    importance = np.ones(20, dtype=np.float32)

    keep = weighted_sample_elimination(points, importance, 10)

    assert keep.sum() == 10
    assert keep[19]  # the isolated point has ~zero crowding weight - never a removal candidate
    assert keep[:19].sum() == 9  # the cluster is thinned down to fill the rest of the budget


def test_low_importance_point_is_eliminated_first_at_equal_density():
    # four coincident points (zero pairwise distance -> maximal, equal geometric weight);
    # only the importance bias breaks the tie
    points = np.zeros((4, 3), dtype=np.float32)
    importance = np.array([1.0, 1.0, 1.0, 0.1], dtype=np.float32)

    keep = weighted_sample_elimination(points, importance, 3)

    assert keep.sum() == 3
    assert not keep[3]  # the low-importance point is the one eliminated


def test_zero_importance_bias_ignores_importance():
    points = np.zeros((4, 3), dtype=np.float32)
    importance = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    # with bias disabled, elimination order is a pure (arbitrary but deterministic) tie-break,
    # not driven by importance - just check it still returns a valid-size mask
    keep = weighted_sample_elimination(points, importance, 3, importance_bias=0.0)
    assert keep.sum() == 3
