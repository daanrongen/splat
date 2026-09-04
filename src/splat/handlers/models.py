from typing import Any

from splat.application.models_admin import (
    ListModelsUseCase,
    ModelInfoUseCase,
    PullModelUseCase,
    RemoveModelUseCase,
)
from splat.registry.wiring import get_model_source


def list_models() -> list[tuple[Any, bool]]:
    return ListModelsUseCase(get_model_source()).execute()


def pull(name: str) -> None:
    PullModelUseCase(get_model_source()).execute(name)


def info(name: str) -> Any:
    return ModelInfoUseCase().execute(name)


def rm(name: str) -> None:
    RemoveModelUseCase(get_model_source()).execute(name)
