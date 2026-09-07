from splat.adapters.formats.image import (
    decode_rgb_or_rgba,
    encode_png,
    read_rgb,
    read_rgb_or_rgba,
    resize,
    write_png,
)
from tests.image_helpers import sample_rgb, sample_rgba


def test_rgb_png_round_trip(tmp_path):
    path = tmp_path / "rgb.png"
    image = sample_rgb((3, 2), color=(10, 20, 30))

    write_png(path, image)

    decoded = read_rgb(path)
    assert decoded.shape == (2, 3, 3)
    assert decoded[0, 0].tolist() == [10, 20, 30]


def test_rgba_png_round_trip_preserves_alpha(tmp_path):
    path = tmp_path / "rgba.png"
    image = sample_rgba((3, 2), color=(10, 20, 30, 128))

    write_png(path, image)

    decoded = read_rgb_or_rgba(path)
    assert decoded.shape == (2, 3, 4)
    assert decoded[0, 0].tolist() == [10, 20, 30, 128]


def test_decode_png_bytes_preserves_channel_order():
    image = sample_rgb((2, 2), color=(5, 10, 15))

    decoded = decode_rgb_or_rgba(encode_png(image))

    assert decoded[0, 0].tolist() == [5, 10, 15]


def test_resize_uses_width_height_size_order():
    image = sample_rgb((3, 2))

    resized = resize(image, (6, 4))

    assert resized.shape == (4, 6, 3)
