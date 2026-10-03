"""Per-view exposure/white-balance correction across a multi-photo capture,
so drift doesn't bake into per-Gaussian SH color as spurious view-dependent
noise ahead of reconstruction.

Matches each image's per-channel mean to the cohort's per-channel median (a
gray-world-style gain correction) - the cheapest per-view color-correction
model in the appearance-embedding/tone-curve literature (NeRF-in-the-Wild,
Luminance-GS-style view-adaptive tone curves), trivially portable to numpy
since it needs no differentiable rasterizer to fit against. Multi-view
gaussian backends run it on their inputs before reconstruction.
"""

import numpy as np


def normalize_exposure(images: list[np.ndarray]) -> list[np.ndarray]:
    channel_means = np.array(
        [image[..., :3].reshape(-1, 3).astype(np.float64).mean(axis=0) for image in images]
    )
    target = np.median(channel_means, axis=0)

    corrected = []
    for image, means in zip(images, channel_means, strict=True):
        safe_means = np.where(means > 1e-6, means, 1.0)
        gain = np.where(means > 1e-6, target / safe_means, 1.0).astype(np.float32)
        rgb = np.clip(image[..., :3].astype(np.float32) * gain, 0, 255).astype(np.uint8)
        corrected.append(np.dstack([rgb, image[..., 3]]) if image.shape[-1] == 4 else rgb)
    return corrected
