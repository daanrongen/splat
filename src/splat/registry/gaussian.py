from dataclasses import dataclass

from splat.domain.value_objects import ModelLicense
from splat.ports.reconstruction import ReconstructionBackend


@dataclass(frozen=True)
class GaussianModelDescriptor:
    name: str
    backend_cls: type[ReconstructionBackend]
    hf_repo_id: str
    license: ModelLicense
    min_images: int
    max_images: int | None


def _build_catalog() -> dict[str, GaussianModelDescriptor]:
    # Imported lazily so a missing/optional adapter dependency can't break
    # every other command — only `splat models pull/gaussian --model ...`
    # needs the reconstruction adapters to actually import cleanly.
    from splat.adapters.reconstruction.mvsplat import MVSplatBackend
    from splat.domain.value_objects import MIT

    return {
        "mvsplat": GaussianModelDescriptor(
            name="mvsplat",
            backend_cls=MVSplatBackend,
            hf_repo_id="dylanebert/mvsplat",
            license=MIT,
            min_images=2,
            max_images=None,
        ),
    }


GAUSSIAN_CATALOG: dict[str, GaussianModelDescriptor] = _build_catalog()
