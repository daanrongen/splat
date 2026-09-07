from dataclasses import dataclass
from typing import Any

from splat.domain.errors import UnsupportedFormat
from splat.ports.model_source import ModelSource
from splat.registry.caption import CAPTION_CATALOG
from splat.registry.depth import DEPTH_CATALOG
from splat.registry.diffuse import DIFFUSION_CATALOG
from splat.registry.embed import EMBEDDING_CATALOG
from splat.registry.gaussian import GAUSSIAN_CATALOG
from splat.registry.segment import SEGMENTATION_CATALOG
from splat.registry.upscale import UPSCALE_CATALOG

# The stage each catalog serves, i.e. the command whose `--model` accepts these
# names. One tuple rather than a merge plus a parallel lookup table, so a new
# catalog cannot be registered without declaring what it is for.
_CATALOGS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("gaussian", GAUSSIAN_CATALOG),
    ("caption", CAPTION_CATALOG),
    ("diffuse", DIFFUSION_CATALOG),
    ("embed", EMBEDDING_CATALOG),
    ("segment", SEGMENTATION_CATALOG),
    ("depth", DEPTH_CATALOG),
    ("upscale", UPSCALE_CATALOG),
)


def _all_catalogs() -> dict[str, Any]:
    return {name: descriptor for _, catalog in _CATALOGS for name, descriptor in catalog.items()}


def model_stage(name: str) -> str:
    """The `splat` command this model can be passed to as `--model`."""
    return next((stage for stage, catalog in _CATALOGS if name in catalog), "-")


def model_sources(descriptor: Any) -> list[str]:
    repo_ids = getattr(descriptor, "hf_repo_ids", None)
    if isinstance(repo_ids, dict):
        return [repo_ids[key] for key in sorted(repo_ids)]
    repo_id = getattr(descriptor, "hf_repo_id", None)
    return [repo_id] if repo_id else []


def model_source_label(descriptor: Any) -> str:
    return ", ".join(model_sources(descriptor)) or "local runtime"


def image_count_range(descriptor: Any) -> tuple[int | None, int | None]:
    """(min, max) images a reconstruction backend accepts, read straight off
    the backend class — `required_image_count` is a classmethod so this
    doesn't need to pull weights or instantiate anything just to display it.
    """
    backend_cls = getattr(descriptor, "backend_cls", None)
    required_image_count = getattr(backend_cls, "required_image_count", None)
    if required_image_count is None:
        return (None, None)
    return required_image_count()


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


@dataclass(frozen=True)
class OrphanedWeights:
    repo_id: str
    size_bytes: int


class PruneModelsUseCase:
    """Cached repos no catalog entry references any more - left behind when a
    model is dropped from the catalog, since `models rm` can only address
    names the catalog still knows."""

    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def find(self) -> list[OrphanedWeights]:
        wanted = {
            source
            for descriptor in _all_catalogs().values()
            for source in model_sources(descriptor)
        }
        return [
            OrphanedWeights(repo_id, self._model_source.size_on_disk(repo_id))
            for repo_id in self._model_source.list_cached()
            if repo_id not in wanted
        ]

    def execute(self) -> list[OrphanedWeights]:
        orphans = self.find()
        for orphan in orphans:
            self._model_source.remove(orphan.repo_id)
        return orphans


class RemoveModelUseCase:
    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def execute(self, name: str) -> None:
        for source in model_sources(_lookup(name)):
            self._model_source.remove(source)
