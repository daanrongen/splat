"""Simple, dependency-free compression: opacity/outlier pruning plus
optional float16 quantization and SH-rest dropping, exposed as two named
presets rather than raw numeric knobs (mirroring SuperSplat's "Compressed
Ply" idiom)."""

from dataclasses import dataclass

import numpy as np

from splat.domain.gaussians import GaussianCloud, GaussianCloudMetadata


@dataclass(frozen=True)
class CompressionProfile:
    opacity_threshold: float
    outlier_std: float | None
    quantize_fp16: bool
    drop_sh_rest: bool


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
    ),
}


class PruneQuantizeCompressor:
    name = "prune-quantize"

    def compress(
        self, cloud: GaussianCloud, *, profile: str = "web-delivery", **params
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
            centroid = cloud.means.mean(axis=0)
            distances = np.linalg.norm(cloud.means - centroid, axis=1)
            mean_dist, std_dist = distances.mean(), distances.std()
            if std_dist > 0:
                keep &= distances <= mean_dist + spec.outlier_std * std_dist

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
            metadata=GaussianCloudMetadata(
                source_format=cloud.metadata.source_format,
                source_model=cloud.metadata.source_model,
                license=cloud.metadata.license,
                up_axis=cloud.metadata.up_axis,
                coordinate_convention=cloud.metadata.coordinate_convention,
            ),
        )
