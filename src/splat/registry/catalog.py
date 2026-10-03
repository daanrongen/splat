from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal

from splat.domain.errors import UnsupportedFormat
from splat.domain.value_objects import ModelLicense

Runtime = Literal["mlx", "coreml", "torch"]


@dataclass(frozen=True)
class ModelDescriptor:
    name: str
    backend: str
    license: ModelLicense
    runtime: Runtime
    hf_repo_ids: tuple[str, ...] = ()
    min_images: int | None = None
    max_images: int | None = None
    dimension: int | None = None
    normalized: bool | None = None
    notes: str = ""
    backend_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def hf_repo_id(self) -> str | None:
        return self.hf_repo_ids[0] if self.hf_repo_ids else None

    @property
    def backend_cls(self) -> type[Any]:
        module_name, class_name = self.backend.rsplit(":", 1)
        return getattr(import_module(module_name), class_name)


def lookup(catalog: dict[str, ModelDescriptor], name: str, label: str = "") -> ModelDescriptor:
    try:
        return catalog[name]
    except KeyError as exc:
        available = ", ".join(sorted(catalog))
        raise UnsupportedFormat(f"Unknown {label}model {name!r}. Available: {available}") from exc
