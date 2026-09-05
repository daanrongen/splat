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
