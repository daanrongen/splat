from typing import Protocol

from splat.domain.gaussians import GaussianCloud


class Declutterer(Protocol):
    name: str

    def declutter(self, cloud: GaussianCloud, **params) -> GaussianCloud: ...
