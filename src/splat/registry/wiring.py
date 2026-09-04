"""The project's entire "dependency injection": plain dict lookups and
factory functions. No container, no decorator-based registration magic —
extending it means adding one dict entry and one adapter file.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from splat.domain.errors import UnsupportedFormat
from splat.domain.value_objects import ModelLicense
from splat.env import resolve as resolve_env
from splat.ports.asset_cache import AssetCache
from splat.ports.caption import CaptioningBackend
from splat.ports.depth import DepthEstimationBackend

if TYPE_CHECKING:
    # Deferred: splat.ports.client imports handlers/*.py, which imports this
    # module — a real cycle if resolved at import time rather than lazily.
    from splat.ports.client import SplatClient
from splat.ports.diffusion import DiffusionBackend
from splat.ports.mesh import MeshPredictionBackend
from splat.ports.model_source import ModelSource
from splat.ports.reconstruction import ReconstructionBackend
from splat.ports.segmentation import SegmentationBackend
from splat.ports.splat_io import SplatReader, SplatWriter
from splat.ports.upscaling import UpscalingBackend
from splat.registry.formats import FORMAT_READERS, FORMAT_WRITERS
from splat.registry.gaussian import GAUSSIAN_CATALOG


def get_reader(ext: str) -> SplatReader:
    try:
        return FORMAT_READERS[ext]()
    except KeyError as exc:
        raise UnsupportedFormat(f"No reader registered for format {ext!r}") from exc


def get_writer(ext: str) -> SplatWriter:
    try:
        return FORMAT_WRITERS[ext]()
    except KeyError as exc:
        raise UnsupportedFormat(f"No writer registered for format {ext!r}") from exc


def is_known_format(ext: str) -> bool:
    return ext in FORMAT_READERS or ext in FORMAT_WRITERS


def get_model_source() -> ModelSource:
    from splat.adapters.model_sources.huggingface import HuggingFaceModelSource

    return HuggingFaceModelSource()


def get_asset_cache() -> AssetCache:
    from splat.adapters.cache.filesystem import FilesystemAssetCache

    return FilesystemAssetCache()


def get_client() -> "SplatClient":
    url = resolve_env("SPLAT_URL", "")
    if url:
        from splat.adapters.client.http import RemoteSplatClient

        return RemoteSplatClient(url)

    from splat.adapters.client.local import LocalSplatClient

    return LocalSplatClient()


def get_reconstruction_backend(
    name: str, *, model_source: ModelSource, device: str = "auto"
) -> ReconstructionBackend:
    try:
        descriptor = GAUSSIAN_CATALOG[name]
    except KeyError as exc:
        available = ", ".join(sorted(GAUSSIAN_CATALOG))
        raise UnsupportedFormat(f"Unknown model {name!r}. Available: {available}") from exc

    weights_path: Path = model_source.pull(descriptor.hf_repo_id)
    return descriptor.backend_cls(
        weights_path=weights_path, device=device, license=descriptor.license
    )


def model_license(name: str) -> ModelLicense:
    return GAUSSIAN_CATALOG[name].license


def get_diffusion_backend(name: str, *, device: str = "auto") -> DiffusionBackend:
    from splat.registry.diffusion import DIFFUSION_CATALOG

    try:
        descriptor = DIFFUSION_CATALOG[name]
    except KeyError as exc:
        available = ", ".join(sorted(DIFFUSION_CATALOG))
        raise UnsupportedFormat(
            f"Unknown diffusion model {name!r}. Available: {available}"
        ) from exc

    return descriptor.backend_cls(
        hf_repo_id=descriptor.hf_repo_id,
        sdxl=descriptor.sdxl,
        license=descriptor.license,
        device=device,
    )


def get_segmentation_backend(name: str, *, device: str = "auto") -> SegmentationBackend:
    from splat.registry.segmentation import SEGMENTATION_CATALOG

    try:
        descriptor = SEGMENTATION_CATALOG[name]
    except KeyError as exc:
        available = ", ".join(sorted(SEGMENTATION_CATALOG))
        raise UnsupportedFormat(
            f"Unknown segmentation model {name!r}. Available: {available}"
        ) from exc

    return descriptor.backend_cls(
        hf_repo_id=descriptor.hf_repo_id, license=descriptor.license, device=device
    )


def get_mesh_backend(name: str, *, device: str = "auto") -> MeshPredictionBackend:
    from splat.registry.mesh import MESH_PREDICTION_CATALOG

    try:
        descriptor = MESH_PREDICTION_CATALOG[name]
    except KeyError as exc:
        available = ", ".join(sorted(MESH_PREDICTION_CATALOG))
        raise UnsupportedFormat(f"Unknown mesh model {name!r}. Available: {available}") from exc

    return descriptor.backend_cls(
        hf_repo_id=descriptor.hf_repo_id, license=descriptor.license, device=device
    )


def get_depth_backend(name: str, *, device: str = "auto") -> DepthEstimationBackend:
    from splat.registry.depth import DEPTH_CATALOG

    try:
        descriptor = DEPTH_CATALOG[name]
    except KeyError as exc:
        available = ", ".join(sorted(DEPTH_CATALOG))
        raise UnsupportedFormat(f"Unknown depth model {name!r}. Available: {available}") from exc

    return descriptor.backend_cls(
        hf_repo_id=descriptor.hf_repo_id, license=descriptor.license, device=device
    )


def get_caption_backend(name: str, *, device: str = "auto") -> CaptioningBackend:
    from splat.registry.caption import CAPTION_CATALOG

    try:
        descriptor = CAPTION_CATALOG[name]
    except KeyError as exc:
        available = ", ".join(sorted(CAPTION_CATALOG))
        raise UnsupportedFormat(f"Unknown caption model {name!r}. Available: {available}") from exc

    return descriptor.backend_cls(
        hf_repo_id=descriptor.hf_repo_id, license=descriptor.license, device=device
    )


def get_upscale_backend(name: str) -> UpscalingBackend:
    from splat.registry.upscale import UPSCALE_CATALOG

    try:
        descriptor = UPSCALE_CATALOG[name]
    except KeyError as exc:
        available = ", ".join(sorted(UPSCALE_CATALOG))
        raise UnsupportedFormat(f"Unknown upscale model {name!r}. Available: {available}") from exc

    return descriptor.backend_cls(license=descriptor.license)
