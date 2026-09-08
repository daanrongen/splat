from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from splat.application.pipeline import run_gaussian, run_render
from splat.domain.contracts import Requirement, StageContract, validate_inputs
from splat.domain.errors import SplatDomainError
from splat.domain.manifest import Manifest, ManifestKind
from splat.registry.wiring import (
    get_manifest_repository,
    get_model_source,
    get_reconstruction_backend,
    get_render_backend,
)


def _single_image_models() -> list[str]:
    """Cataloged backends that reconstruct from one image. Derived rather than
    hardcoded so the hint below cannot go stale the way it did when `sharp`
    landed and the multi-view hint kept telling people to shoot more frames."""
    from splat.application.models_admin import image_count_range
    from splat.registry.gaussian import GAUSSIAN_CATALOG

    return sorted(
        name
        for name, descriptor in GAUSSIAN_CATALOG.items()
        if image_count_range(descriptor)[0] == 1
    )


def _input_hint(min_images: int) -> str:
    if min_images <= 1:
        return (
            "gaussian has no text-to-3D or depth-only reconstruction path; pipe an "
            "image or sticker asset in."
        )
    hint = (
        "gaussian has no text-to-3D or depth-only reconstruction path; pipe 3+ image "
        "assets of the same scene from different viewpoints, e.g. "
        "`splat gaussian frame-*.png`."
    )
    single = _single_image_models()
    if single:
        hint += f" To reconstruct from one image, use --model {' or '.join(single)}."
    return hint


@dataclass(frozen=True)
class GaussianRequest:
    inputs: list[Manifest]
    model: str = "sharp"
    device: str = "auto"
    quality: str = "fast"
    iters: int | None = None
    max_dim: int | None = None
    sh_degree: int | None = None
    poses: str = "auto"
    refine_poses: str = "auto"
    low_memory: bool = False
    seed: int = 0
    focal_35mm: float = 30.0
    orbit_frames: int | None = None
    orbit_degrees: float = 30.0


def _render_orbit(
    cloud_asset: Manifest, *, orbit_frames: int, orbit_degrees: float
) -> list[Manifest]:
    """Renders `orbit_frames` synthetic views of a just-reconstructed cloud,
    swept across `orbit_degrees` of azimuth - self-consistent by construction
    since every frame comes from the same Gaussians, unlike prompting a
    diffusion model for multiple viewpoints (see #95)."""
    if orbit_frames < 2:
        raise SplatDomainError("--orbit-frames must be at least 2 to sweep an angle range.")

    cache = get_manifest_repository()
    backend = get_render_backend()
    offsets = np.linspace(-orbit_degrees / 2, orbit_degrees / 2, orbit_frames)
    return [
        run_render(
            backend,
            cache,
            model_name="blender",
            input_asset=cloud_asset,
            params={
                "width": 1280,
                "height": 720,
                "samples": 32,
                "engine": "cycles",
                "background": "black",
                "azimuth": float(azimuth),
                "elevation": None,
                "distance": None,
                "fov": None,
                "look_at": None,
            },
        )
        for azimuth in offsets
    ]


def handle(
    request: GaussianRequest, *, on_progress: Callable[[str], None] | None = None
) -> list[Manifest]:
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
                hint=_input_hint(min_images),
            ),
        ),
        produces=ManifestKind.GAUSSIAN_CLOUD,
    )
    validate_inputs(contract, request.inputs)

    cache = get_manifest_repository()
    params = {
        "device": request.device,
        "quality": request.quality,
        "poses": request.poses,
        "refine_poses": request.refine_poses,
        "low_memory": request.low_memory,
        "seed": request.seed,
        "focal_35mm": request.focal_35mm,
    }
    if request.iters is not None:
        params["iters"] = request.iters
    if request.max_dim is not None:
        params["max_dim"] = request.max_dim
    if request.sh_degree is not None:
        params["sh_degree"] = request.sh_degree

    cloud = run_gaussian(
        backend,
        cache,
        model_name=request.model,
        input_assets=request.inputs,
        params=params,
        on_progress=on_progress,
    )
    results = [cloud]
    if request.orbit_frames is not None:
        results.extend(
            _render_orbit(
                cloud, orbit_frames=request.orbit_frames, orbit_degrees=request.orbit_degrees
            )
        )
    return results
