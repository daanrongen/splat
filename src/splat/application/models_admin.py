from collections.abc import Callable
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


def model_license(created_by: str) -> str:
    """SPDX id of the catalogued model behind a `<stage>:<model>` created_by, or ''."""
    descriptor = _all_catalogs().get(created_by.partition(":")[2])
    return descriptor.license.spdx_id if descriptor else ""


def model_source_label(descriptor: ModelDescriptor) -> str:
    return ", ".join(descriptor.hf_repo_ids) or "local runtime"


def _lookup(name: str) -> ModelDescriptor:
    return lookup(_all_catalogs(), name)


@dataclass(frozen=True)
class ModelStatus:
    name: str
    stage: str
    runtime: str
    license: str  # SPDX id only; `commercial` says whether use is restricted
    commercial: bool
    cached: bool
    size_bytes: int
    managed: bool = True


class ListModelsUseCase:
    """The catalog with what is installed and how big it is, then every store directory
    no catalog entry uses."""

    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def _catalog_row(self, descriptor: ModelDescriptor) -> ModelStatus:
        source = self._model_source
        return ModelStatus(
            name=descriptor.name,
            stage=model_stage(descriptor.name),
            runtime=descriptor.runtime,
            license=descriptor.license.spdx_id,
            commercial=descriptor.license.is_commercial,
            cached=all(
                source.is_cached(repo, descriptor.files(repo)) for repo in descriptor.hf_repo_ids
            ),
            size_bytes=sum(source.size_on_disk(repo) for repo in descriptor.hf_repo_ids),
        )

    def execute(self) -> list[ModelStatus]:
        catalog = [self._catalog_row(descriptor) for descriptor in _all_catalogs().values()]
        wanted = {
            repo for descriptor in _all_catalogs().values() for repo in descriptor.hf_repo_ids
        }
        unmanaged = [
            ModelStatus(
                name=name,
                stage="-",
                runtime="-",
                license="-",
                commercial=True,
                cached=True,
                size_bytes=self._model_source.size_on_disk(name),
                managed=False,
            )
            for name in self._model_source.list_cached()
            if name not in wanted
        ]
        return catalog + unmanaged


def _for_repo(
    repo_id: str, on_progress: Callable[[str, int, int], None] | None
) -> Callable[[int, int], None] | None:
    return None if on_progress is None else lambda done, total: on_progress(repo_id, done, total)


class PullModelUseCase:
    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def execute(
        self, name: str, on_progress: Callable[[str, int, int], None] | None = None
    ) -> None:
        descriptor = _lookup(name)
        for source in descriptor.hf_repo_ids:
            self._model_source.pull(
                source, files=descriptor.files(source), on_progress=_for_repo(source, on_progress)
            )


class ModelInfoUseCase:
    def execute(self, name: str) -> ModelDescriptor:
        return _lookup(name)


@dataclass(frozen=True)
class OrphanedWeights:
    repo_id: str
    size_bytes: int
    legacy: bool = False


class PruneModelsUseCase:
    """Directories in the model store no catalog entry uses (left behind when a model is
    dropped or an older layout changes), and hub-cache copies of catalogued repos the
    store already holds in full."""

    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def find(self) -> list[OrphanedWeights]:
        files = {
            source: descriptor.files(source)
            for descriptor in _all_catalogs().values()
            for source in descriptor.hf_repo_ids
        }
        source = self._model_source
        unmanaged = [
            OrphanedWeights(repo_id, source.size_on_disk(repo_id))
            for repo_id in source.list_cached()
            if repo_id not in files
        ]
        redundant = [
            OrphanedWeights(repo_id, size, legacy=True)
            for repo_id, size in source.list_legacy().items()
            if repo_id in files and source.is_cached(repo_id, files[repo_id])
        ]
        return unmanaged + redundant

    def execute(self) -> list[OrphanedWeights]:
        orphans = self.find()
        for orphan in orphans:
            remove = (
                self._model_source.remove_legacy if orphan.legacy else self._model_source.remove
            )
            remove(orphan.repo_id)
        return orphans


class RemoveModelUseCase:
    def __init__(self, model_source: ModelSource) -> None:
        self._model_source = model_source

    def execute(self, name: str) -> None:
        for source in _lookup(name).hf_repo_ids:
            self._model_source.remove(source)
