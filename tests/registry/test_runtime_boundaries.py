import ast
import importlib.util
import sys
from pathlib import Path

import pytest

import splat
from splat.application.models_admin import _all_catalogs
from splat.domain.errors import SplatDomainError
from splat.domain.value_objects import MIT
from splat.registry.catalog import ModelDescriptor

RUNTIME_PACKAGES = {
    "mlx": {"mlx", "mlx3d", "realesrgan_mlx"},
    "coreml": {"coremltools"},
    "torch": {"torch", "torchvision", "transformers", "timm", "open_clip"},
}
SHARED = {"numpy", "PIL", "huggingface_hub", "splat"} | set(sys.stdlib_module_names)
HEAVY = {"cv2", "scipy", "trimesh", "open3d", "huggingface_hub"}.union(*RUNTIME_PACKAGES.values())
ROOT = Path(splat.__file__).parent


def imported_packages(nodes: list[ast.AST]) -> set[str]:
    found = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add((node.module or "").split(".")[0])
    return found


def backend_source(descriptor: ModelDescriptor) -> Path:
    return Path(importlib.util.find_spec(descriptor.backend.split(":")[0]).origin)


@pytest.mark.parametrize("descriptor", _all_catalogs().values(), ids=lambda d: d.name)
def test_backend_imports_only_its_own_runtime(descriptor):
    tree = ast.parse(backend_source(descriptor).read_text())
    allowed = SHARED | RUNTIME_PACKAGES[descriptor.runtime]

    assert imported_packages(list(ast.walk(tree))) - allowed == set()


def test_no_runtime_is_imported_at_module_level_outside_adapters():
    offenders = {}
    for path in ROOT.rglob("*.py"):
        if "adapters" in path.relative_to(ROOT).parts:
            continue
        leaked = imported_packages(ast.parse(path.read_text()).body) & HEAVY
        if leaked:
            offenders[str(path.relative_to(ROOT))] = sorted(leaked)

    assert offenders == {}


def test_failed_runtime_import_names_the_model_and_runtime():
    descriptor = ModelDescriptor(
        name="broken",
        backend="no_such_runtime_module:Backend",
        license=MIT,
        runtime="torch",
    )

    with pytest.raises(SplatDomainError, match=r"'broken'.*torch runtime"):
        _ = descriptor.backend_cls
