from dataclasses import dataclass

from splat.ports.model_source import ModelSource
from splat.registry.caption import CAPTION_CATALOG
from splat.registry.catalog import ModelDescriptor, lookup
from splat.registry.depth import DEPTH_CATALOG
from splat.registry.diffuse import DIFFUSION_CATALOG
from splat.registry.embed import EMBEDDING_CATALOG
from splat.registry.gaussian import GAUSSIAN_CATALOG
from splat.registry.segment import SEGMENTATION_CATALOG
from splat.registry.upscale import UPSCALE_CATALOG

# The stage each catalog serves, i.e. the command whose `--model` accepts these
# names. One tuple rather than a merge plus a parallel lookup table, so a new
# catalog cannot be registered without declaring what it is for.
_CATALOGS: tuple[tuple[str, dict[str, ModelDescriptor]], ...] = (
    ("gaussian", GAUSSIAN_CATALOG),
    ("caption", CAPTION_CATALOG),
    ("diffuse", DIFFUSION_CATALOG),
    ("embed", EMBEDDING_CATALOG),
    ("segment", SEGMENTATION_CATALOG),
    ("depth", DEPTH_CATALOG),
    ("upscale", UPSCALE_CATALOG),
)


def _all_catalogs() -> dict[str, ModelDescriptor]:
    return {name: descriptor for _, catalog in _CATALOGS for name, descriptor in catalog.items()}


def model_stage(name: str) -> str:
    """The `splat` command this model can be passed to as `--model`."""
    return next((stage for stage, catalog in _CATALOGS if name in catalog), "-")


def model_source_label(descriptor: ModelDescriptor) -> str:
    return ", ".join(descriptor.hf_repo_ids) or "local runtime"


def _lookup(name: str) -> ModelDescriptor:
    return lookup(_all_catalogs(), name)


class ListModelsUseCase:
    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def execute(self) -> list[tuple[ModelDescriptor, bool]]:
        return [
            (
                descriptor,
                all(self._model_source.is_cached(source) for source in descriptor.hf_repo_ids),
            )
            for descriptor in _all_catalogs().values()
        ]


class PullModelUseCase:
    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def execute(self, name: str) -> None:
        for source in _lookup(name).hf_repo_ids:
            self._model_source.pull(source)


class ModelInfoUseCase:
    def execute(self, name: str) -> ModelDescriptor:
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
            source for descriptor in _all_catalogs().values() for source in descriptor.hf_repo_ids
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
        for source in _lookup(name).hf_repo_ids:
            self._model_source.remove(source)
