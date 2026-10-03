"""Floater and degeneracy stats for a Gaussian cloud, and fidelity scores of a
render against the photo it should reproduce."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from splat.domain.gaussians import GaussianCloud, to_convention

_TRANSPARENT = 0.05
_NEEDLE_ANISOTROPY = 10.0
_SSIM_WINDOW = 7


def cloud_stats(cloud: GaussianCloud) -> dict:
    opacity = cloud.to_activated_opacities()
    scales = cloud.to_linear_scales()
    anisotropy = scales.max(axis=1) / scales.min(axis=1).clip(1e-12)
    histogram, _ = np.histogram(opacity, bins=5, range=(0.0, 1.0))
    return {
        "opacity_histogram": (histogram / max(len(opacity), 1)).round(4).tolist(),
        "transparent_ratio": round(float((opacity < _TRANSPARENT).mean()), 4),
        "needle_ratio": round(float((anisotropy > _NEEDLE_ANISOTROPY).mean()), 4),
        "out_of_view_ratio": _out_of_view_ratio(cloud),
    }


def _out_of_view_ratio(cloud: GaussianCloud) -> float | None:
    """Share of Gaussians no source camera sees: floaters behind or beside the capture."""
    cameras = cloud.metadata.source_cameras
    if not cameras:
        return None
    colmap = to_convention(cloud, "colmap")
    means = colmap.means.astype(np.float64)
    seen = np.zeros(len(means), dtype=bool)
    for camera in colmap.metadata.source_cameras:
        fx, fy, cx, cy, width, height = camera["intrinsics"]
        local = (means - camera["position"]) @ np.asarray(camera["rotation"]).T
        z = local[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = fx * local[:, 0] / z + cx, fy * local[:, 1] / z + cy
        seen |= (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return round(float(1.0 - seen.mean()), 4)


def image_scores(rendered: np.ndarray, reference: np.ndarray) -> dict:
    """PSNR and SSIM of two same-size uint8 RGB images."""
    a, b = rendered.astype(np.float64) / 255.0, reference.astype(np.float64) / 255.0
    mse = float(((a - b) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
    return {"psnr": round(float(psnr), 2), "ssim": round(_ssim(a, b), 4)}


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over 7x7 uniform windows per channel (scikit-image's default)."""
    c1, c2 = 0.01**2, 0.03**2

    side = min(_SSIM_WINDOW, *a.shape[:2])

    def mean(x: np.ndarray) -> np.ndarray:
        window = (side, side)
        return sliding_window_view(x, window, axis=(0, 1)).mean(axis=(-2, -1))

    mu_a, mu_b = mean(a), mean(b)
    var_a, var_b = mean(a * a) - mu_a**2, mean(b * b) - mu_b**2
    cov = mean(a * b) - mu_a * mu_b
    ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return float(ssim.mean())
