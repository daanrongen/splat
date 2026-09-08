"""Simple, dependency-free compression: opacity/outlier pruning plus
optional float16 quantization and SH-rest dropping, exposed as two named
presets rather than raw numeric knobs (mirroring SuperSplat's "Compressed
Ply" idiom)."""

from dataclasses import dataclass, replace

import numpy as np

from splat.adapters.compression.blue_noise import weighted_sample_elimination
from splat.domain.gaussians import GaussianCloud


@dataclass(frozen=True)
class CompressionProfile:
    opacity_threshold: float
    outlier_std: float | None
    quantize_fp16: bool
    drop_sh_rest: bool
    dedupe_near_identical: bool = False


PROFILES: dict[str, CompressionProfile] = {
    "web-delivery": CompressionProfile(
        opacity_threshold=1.0 / 255.0,
        outlier_std=3.0,
        quantize_fp16=True,
        drop_sh_rest=True,
    ),
    "archival": CompressionProfile(
        opacity_threshold=0.0,
        outlier_std=None,
        quantize_fp16=False,
        drop_sh_rest=False,
        # The one thing this profile can remove without any loss: Gaussians
        # left effectively coincident by multi-view reconstruction overlap.
        dedupe_near_identical=True,
    ),
}


def _dedupe_mask(means: np.ndarray) -> np.ndarray:
    """Keeps the first point in each occupied cell of a grid fine enough
    (1e-5 of the cloud's bounding-box diagonal) to only ever merge points
    that are duplicates in every practical sense, not genuinely distinct
    ones."""
    diagonal = float(np.linalg.norm(means.max(axis=0) - means.min(axis=0)))
    eps = max(diagonal * 1e-5, 1e-6)
    keys = np.round(means / eps).astype(np.int64)
    _, first_idx = np.unique(keys, axis=0, return_index=True)
    mask = np.zeros(len(means), dtype=bool)
    mask[first_idx] = True
    return mask


class PruneQuantizeCompressor:
    name = "prune-quantize"

    def compress(
        self,
        cloud: GaussianCloud,
        *,
        profile: str = "web-delivery",
        pruning: str = "threshold",
        target_count: int | None = None,
        **params,
    ) -> GaussianCloud:
        try:
            spec = PROFILES[profile]
        except KeyError as exc:
            available = ", ".join(sorted(PROFILES))
            raise ValueError(
                f"Unknown compression profile {profile!r}. Available: {available}"
            ) from exc

        keep = np.ones(cloud.point_count, dtype=bool)

        if spec.opacity_threshold > 0:
            keep &= cloud.to_activated_opacities() >= spec.opacity_threshold

        if spec.outlier_std is not None and cloud.point_count > 1:
            centroid = np.median(cloud.means, axis=0)
            distances = np.linalg.norm(cloud.means - centroid, axis=1)
            mean_dist, std_dist = distances.mean(), distances.std()
            if std_dist > 0:
                keep &= distances <= mean_dist + spec.outlier_std * std_dist

        if spec.dedupe_near_identical and cloud.point_count > 1:
            keep &= _dedupe_mask(cloud.means)

        if pruning == "blue-noise":
            if target_count is None:
                raise ValueError("pruning='blue-noise' requires target_count (--target-count).")
            kept_idx = np.flatnonzero(keep)
            survives = weighted_sample_elimination(
                cloud.means[kept_idx], cloud.to_activated_opacities()[kept_idx], target_count
            )
            keep = np.zeros(cloud.point_count, dtype=bool)
            keep[kept_idx[survives]] = True
        elif pruning != "threshold":
            raise ValueError(
                f"Unknown pruning strategy {pruning!r}. Available: threshold, blue-noise"
            )

        sh_rest = cloud.sh_rest[keep] if cloud.sh_rest is not None else None
        sh_degree = cloud.sh_degree
        if spec.drop_sh_rest:
            sh_rest = None
            sh_degree = 0

        means, scales, rotations, opacities, sh_dc = (
            cloud.means[keep],
            cloud.scales[keep],
            cloud.rotations[keep],
            cloud.opacities[keep],
            cloud.sh_dc[keep],
        )

        if spec.quantize_fp16:
            scales = scales.astype(np.float16).astype(np.float32)
            opacities = opacities.astype(np.float16).astype(np.float32)
            sh_dc = sh_dc.astype(np.float16).astype(np.float32)
            if sh_rest is not None:
                sh_rest = sh_rest.astype(np.float16).astype(np.float32)

        return GaussianCloud(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
            sh_degree=sh_degree,
            scale_activation=cloud.scale_activation,
            opacity_activation=cloud.opacity_activation,
            # GaussianCloud.__post_init__ recomputes point_count/sh_degree/
            # activations from the arrays above; replace() carries everything
            # else (camera pose, license, convention, metric_scale) forward
            # instead of silently dropping it.
            metadata=replace(cloud.metadata),
        )
