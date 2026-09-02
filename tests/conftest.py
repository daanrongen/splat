import numpy as np
import pytest

from splat.domain.gaussians import GaussianCloud


@pytest.fixture
def synthetic_cloud() -> GaussianCloud:
    rng = np.random.default_rng(42)
    n = 50
    return GaussianCloud(
        means=rng.uniform(-5, 5, size=(n, 3)).astype(np.float32),
        scales=rng.uniform(-4, -1, size=(n, 3)).astype(np.float32),  # log-scale
        rotations=_random_quaternions(rng, n),
        opacities=rng.uniform(-4, 4, size=n).astype(np.float32),  # logit
        sh_dc=rng.uniform(-1, 1, size=(n, 3)).astype(np.float32),
        sh_rest=rng.uniform(-0.5, 0.5, size=(n, 15, 3)).astype(np.float32),
        sh_degree=3,
    )


def _random_quaternions(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q
