import mlx.core as mx
import numpy as np

from splat.adapters.segment.mlx_sam import _mlx_weights


def _save(tmp_path, **tensors):
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), {k: mx.array(v) for k, v in tensors.items()}
    )


def test_encoder_convs_move_channels_last(tmp_path):
    _save(tmp_path, **{"vision_encoder.neck.conv1.weight": np.zeros((2, 3, 4, 5), np.float32)})

    weights = _mlx_weights(tmp_path)

    assert weights["vision_encoder.neck.conv1.weight"].shape == (2, 4, 5, 3)


def test_decoder_upscale_convs_use_their_own_permutation(tmp_path):
    _save(tmp_path, **{"mask_decoder.upscale_conv1.weight": np.zeros((1, 2, 3, 4), np.float32)})

    weights = _mlx_weights(tmp_path)

    assert weights["mask_decoder.upscale_conv1.weight"].shape == (2, 3, 4, 1)


def test_other_weights_are_left_alone(tmp_path):
    _save(tmp_path, **{"mask_decoder.iou_head.weight": np.zeros((2, 3, 4, 5), np.float32)})

    weights = _mlx_weights(tmp_path)

    assert weights["mask_decoder.iou_head.weight"].shape == (2, 3, 4, 5)
