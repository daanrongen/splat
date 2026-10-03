from splat.domain.value_objects import APPLE_AMLR, MIT
from splat.registry.catalog import ModelDescriptor

GAUSSIAN_CATALOG = {
    "mlx3d-capture": ModelDescriptor(
        name="mlx3d-capture",
        backend="splat.adapters.gaussian.mlx3d_capture:MLX3DCaptureBackend",
        license=MIT,
        runtime="mlx",
        min_images=3,
        notes=(
            "Local Apple Silicon backend using mlx3d's optimization-based capture "
            "pipeline; requires 3+ photos or frames."
        ),
    ),
    "sharp": ModelDescriptor(
        name="sharp",
        backend="splat.adapters.gaussian.sharp:SharpBackend",
        hf_repo_ids=("apple/Sharp",),
        license=APPLE_AMLR,
        runtime="torch",
        min_images=1,
        max_images=1,
        notes=(
            "Apple SHARP: single-image feed-forward 3DGS in one pass. Metric "
            "absolute scale, OpenCV/COLMAP convention. Research use only."
        ),
    ),
}
