from collections.abc import Callable

from splat.application.models_admin import (
    ListModelsUseCase,
    ModelInfoUseCase,
    ModelStatus,
    OrphanedWeights,
    PruneModelsUseCase,
    PullModelUseCase,
    RemoveModelUseCase,
)
from splat.registry.wiring import get_model_source


def list_models() -> list[ModelStatus]:
    return ListModelsUseCase(get_model_source()).execute()


def pull(name: str, *, on_progress: Callable[[str, int, int], None] | None = None) -> None:
    PullModelUseCase(get_model_source()).execute(name, on_progress)


def info(name: str):
    return ModelInfoUseCase().execute(name)


def rm(name: str) -> None:
    RemoveModelUseCase(get_model_source()).execute(name)


def prune(*, apply: bool = False) -> list[OrphanedWeights]:
    use_case = PruneModelsUseCase(get_model_source())
    return use_case.execute() if apply else use_case.find()
