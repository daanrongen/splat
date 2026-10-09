import mlx.core as mx

from splat.adapters.segment._vendor.mlx_sam.utils.amg import batched_mask_to_box


def test_batched_mask_to_box_empty_batch():
    boxes = batched_mask_to_box(mx.zeros((0, 8, 8), dtype=mx.bool_))
    assert boxes.shape == (0, 4)
