class SplatDomainError(Exception):
    """Base class for all domain-level errors."""


class InvalidGaussianCloud(SplatDomainError):
    """A GaussianCloud failed its invariants."""


class UnsupportedSHDegree(SplatDomainError):
    """A spherical-harmonics degree outside the supported range (0-3) was requested."""


class UnsupportedFormat(SplatDomainError):
    """No reader/writer is registered for a given file extension."""


class NonCommercialModelError(SplatDomainError):
    """A model licensed for non-commercial use only was invoked without acknowledgement."""


class WrongManifestKind(SplatDomainError):
    """A stage received manifest(s) whose kind doesn't satisfy its contract."""


class WrongManifestCount(SplatDomainError):
    """A stage received a number of manifests outside its contract's min/max."""


class RenderBackendError(SplatDomainError):
    """A render backend's external process (e.g. Blender) was missing or failed."""


class ReconstructionBackendError(SplatDomainError):
    """A reconstruction backend's weights, device or input were unusable."""


class ManifestHasChildren(SplatDomainError):
    """A manifest other manifests derive from was asked to be deleted on its own."""


class ContractViolation(SplatDomainError):
    """A stage output does not have its kind's canonical form."""
