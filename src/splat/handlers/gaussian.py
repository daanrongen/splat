from dataclasses import dataclass

from splat.application.pipeline import run_gaussian
from splat.domain.asset import Asset, AssetKind
from splat.domain.errors import SplatDomainError
from splat.registry.wiring import get_asset_cache, get_model_source, get_reconstruction_backend


@dataclass(frozen=True)
class GaussianRequest:
    inputs: list[Asset]
    model: str = "mlx3d-capture"
    device: str = "auto"
    quality: str = "fast"
    iters: int | None = None
    max_dim: int | None = None
    sh_degree: int | None = None
    poses: str = "auto"
    refine_poses: str = "auto"
    low_memory: bool = False
    seed: int = 0


def handle(request: GaussianRequest) -> list[Asset]:
    model_source = get_model_source()
    backend = get_reconstruction_backend(
        request.model, model_source=model_source, device=request.device
    )

    for asset in request.inputs:
        if asset.kind not in (AssetKind.IMAGE, AssetKind.STICKER):
            raise SplatDomainError(
                f"Gaussian reconstruction requires image/sticker assets, got {asset.kind.value}."
            )

    min_images, max_images = backend.required_image_count()
    if len(request.inputs) < min_images or (
        max_images is not None and len(request.inputs) > max_images
    ):
        upper = max_images if max_images is not None else "∞"
        raise SplatDomainError(
            f"Model {request.model!r} requires between {min_images} and {upper} images, "
            f"got {len(request.inputs)}."
        )

    cache = get_asset_cache()
    params = {
        "device": request.device,
        "quality": request.quality,
        "poses": request.poses,
        "refine_poses": request.refine_poses,
        "low_memory": request.low_memory,
        "seed": request.seed,
    }
    if request.iters is not None:
        params["iters"] = request.iters
    if request.max_dim is not None:
        params["max_dim"] = request.max_dim
    if request.sh_degree is not None:
        params["sh_degree"] = request.sh_degree

    return [
        run_gaussian(
            backend,
            cache,
            model_name=request.model,
            input_assets=request.inputs,
            params=params,
        )
    ]
