from splat.domain.errors import UnsupportedFormat
from splat.ports.model_source import ModelSource
from splat.registry.models import MODEL_CATALOG, ModelDescriptor


def _lookup(name: str) -> ModelDescriptor:
    try:
        return MODEL_CATALOG[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_CATALOG))
        raise UnsupportedFormat(f"Unknown model {name!r}. Available: {available}") from exc


class ListModelsUseCase:
    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def execute(self) -> list[tuple[ModelDescriptor, bool]]:
        return [
            (descriptor, self._model_source.is_cached(descriptor.hf_repo_id))
            for descriptor in MODEL_CATALOG.values()
        ]


class PullModelUseCase:
    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def execute(self, name: str) -> None:
        self._model_source.pull(_lookup(name).hf_repo_id)


class ModelInfoUseCase:
    def execute(self, name: str) -> ModelDescriptor:
        return _lookup(name)


class RemoveModelUseCase:
    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def execute(self, name: str) -> None:
        self._model_source.remove(_lookup(name).hf_repo_id)
