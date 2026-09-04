from typing import Any

from splat.domain.errors import UnsupportedFormat
from splat.ports.model_source import ModelSource
from splat.registry.caption import CAPTION_CATALOG
from splat.registry.depth import DEPTH_CATALOG
from splat.registry.diffusion import DIFFUSION_CATALOG
from splat.registry.embedding import EMBEDDING_CATALOG
from splat.registry.gaussian import GAUSSIAN_CATALOG
from splat.registry.mesh import MESH_PREDICTION_CATALOG
from splat.registry.segmentation import SEGMENTATION_CATALOG
from splat.registry.upscale import UPSCALE_CATALOG


def _all_catalogs() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for catalog in (
        GAUSSIAN_CATALOG,
        CAPTION_CATALOG,
        DIFFUSION_CATALOG,
        EMBEDDING_CATALOG,
        SEGMENTATION_CATALOG,
        DEPTH_CATALOG,
        MESH_PREDICTION_CATALOG,
        UPSCALE_CATALOG,
    ):
        merged.update(catalog)
    return merged


def model_sources(descriptor: Any) -> list[str]:
    repo_ids = getattr(descriptor, "hf_repo_ids", None)
    if isinstance(repo_ids, dict):
        return [repo_ids[key] for key in sorted(repo_ids)]
    return [descriptor.hf_repo_id]


def model_source_label(descriptor: Any) -> str:
    return ", ".join(model_sources(descriptor))


def _lookup(name: str) -> Any:
    catalog = _all_catalogs()
    try:
        return catalog[name]
    except KeyError as exc:
        available = ", ".join(sorted(catalog))
        raise UnsupportedFormat(f"Unknown model {name!r}. Available: {available}") from exc


class ListModelsUseCase:
    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def execute(self) -> list[tuple[Any, bool]]:
        return [
            (
                descriptor,
                all(self._model_source.is_cached(source) for source in model_sources(descriptor)),
            )
            for descriptor in _all_catalogs().values()
        ]


class PullModelUseCase:
    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def execute(self, name: str) -> None:
        for source in model_sources(_lookup(name)):
            self._model_source.pull(source)


class ModelInfoUseCase:
    def execute(self, name: str) -> Any:
        return _lookup(name)


class RemoveModelUseCase:
    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def execute(self, name: str) -> None:
        for source in model_sources(_lookup(name)):
            self._model_source.remove(source)
