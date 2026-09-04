from pathlib import Path

from splat.application.inspect import InfoUseCase, ValidateUseCase, ValidationResult
from splat.domain.gaussians import GaussianCloud
from splat.registry.wiring import get_reader


def info(path: Path) -> GaussianCloud:
    return InfoUseCase(get_reader(path.suffix)).execute(path)


def validate(path: Path, *, strict: bool = False) -> ValidationResult:
    return ValidateUseCase(get_reader(path.suffix)).execute(path, strict=strict)
