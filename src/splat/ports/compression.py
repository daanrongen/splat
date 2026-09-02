from typing import Protocol

from splat.domain.gaussians import GaussianCloud


class Compressor(Protocol):
    name: str

    def compress(
        self, cloud: GaussianCloud, *, profile: str = "web-delivery", **params
    ) -> GaussianCloud: ...
