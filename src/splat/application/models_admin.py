from typing import Any

from splat.domain.errors import UnsupportedFormat
from splat.ports.model_source import ModelSource
from splat.registry.depth import DEPTH_CATALOG
from splat.registry.gaussian import GAUSSIAN_CATALOG
from splat.registry.generation import GENERATION_CATALOG
from splat.registry.mesh import MESH_CATALOG, MESH_PREDICTION_CATALOG
from splat.registry.segmentation import SEGMENTATION_CATALOG


def _all_catalogs() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for catalog in (
        GAUSSIAN_CATALOG,
        GENERATION_CATALOG,
        SEGMENTATION_CATALOG,
        DEPTH_CATALOG,
        MESH_CATALOG,
        MESH_PREDICTION_CATALOG,
    ):
        merged.update(catalog)
    return merged


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
            (descriptor, self._model_source.is_cached(descriptor.hf_repo_id))
            for descriptor in _all_catalogs().values()
        ]


class PullModelUseCase:
    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def execute(self, name: str) -> None:
        self._model_source.pull(_lookup(name).hf_repo_id)


class ModelInfoUseCase:
    def execute(self, name: str) -> Any:
        return _lookup(name)


class RemoveModelUseCase:
    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def execute(self, name: str) -> None:
        self._model_source.remove(_lookup(name).hf_repo_id)
