from dataclasses import dataclass

from splat.application.pipeline import run_gaussian
from splat.domain.contracts import Requirement, StageContract, validate_inputs
from splat.domain.manifest import Manifest, ManifestKind
from splat.registry.wiring import get_asset_cache, get_model_source, get_reconstruction_backend


@dataclass(frozen=True)
class GaussianRequest:
    inputs: list[Manifest]
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


def handle(request: GaussianRequest) -> list[Manifest]:
    model_source = get_model_source()
    backend = get_reconstruction_backend(
        request.model, model_source=model_source, device=request.device
    )

    # Counts are model-specific (backend.required_image_count()), so this
    # contract is built per-request instead of living as a static registry
    # constant like every other stage's.
    min_images, max_images = backend.required_image_count()
    contract = StageContract(
        stage=f"gaussian ({request.model})",
        inputs=(
            Requirement(
                name="images",
                any_of_tags=frozenset({"colorlike"}),
                min_count=min_images,
                max_count=max_images,
                hint=(
                    "gaussian has no text-to-3D or depth-only reconstruction path — pipe "
                    "image/sticker assets in instead, e.g. "
                    "`splat diffuse ... | splat segment - | splat gaussian -`."
                ),
            ),
        ),
        produces=ManifestKind.GAUSSIAN_CLOUD,
    )
    validate_inputs(contract, request.inputs)

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
