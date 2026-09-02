"""The project's entire "dependency injection": plain dict lookups and
factory functions. No container, no decorator-based registration magic —
extending it means adding one dict entry and one adapter file.
"""

from pathlib import Path

from splat.domain.errors import UnsupportedFormat
from splat.domain.value_objects import ModelLicense
from splat.ports.model_source import ModelSource
from splat.ports.reconstruction import ReconstructionBackend
from splat.ports.splat_io import SplatReader, SplatWriter
from splat.registry.formats import FORMAT_READERS, FORMAT_WRITERS
from splat.registry.models import MODEL_CATALOG


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


def get_reconstruction_backend(
    name: str, *, model_source: ModelSource, device: str = "auto"
) -> ReconstructionBackend:
    try:
        descriptor = MODEL_CATALOG[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_CATALOG))
        raise UnsupportedFormat(f"Unknown model {name!r}. Available: {available}") from exc

    weights_path: Path = model_source.pull(descriptor.hf_repo_id)
    return descriptor.backend_cls(
        weights_path=weights_path, device=device, license=descriptor.license
    )


def model_license(name: str) -> ModelLicense:
    return MODEL_CATALOG[name].license
