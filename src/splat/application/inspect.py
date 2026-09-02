from dataclasses import dataclass
from pathlib import Path

from splat.domain.gaussians import GaussianCloud
from splat.ports.splat_io import SplatReader


class InfoUseCase:
    def __init__(self, reader: SplatReader) -> None:
        self._reader = reader

    def execute(self, path: Path) -> GaussianCloud:
        return self._reader.read(path)


@dataclass
class ValidationResult:
    cloud: GaussianCloud
    issues: list[str]


class ValidateUseCase:
    """GaussianCloud construction already enforces the core invariants
    (raises InvalidGaussianCloud); --strict adds extra sanity checks that
    are valid-but-suspicious rather than structurally broken."""

    def __init__(self, reader: SplatReader) -> None:
        self._reader = reader

    def execute(self, path: Path, *, strict: bool = False) -> ValidationResult:
        cloud = self._reader.read(path)
        issues: list[str] = []
        if strict:
            if (cloud.to_linear_scales() <= 0).any():
                issues.append("non-positive scale values present")
            activated = cloud.to_activated_opacities()
            if ((activated < 0) | (activated > 1)).any():
                issues.append("opacity outside [0, 1] after activation")
        return ValidationResult(cloud=cloud, issues=issues)
