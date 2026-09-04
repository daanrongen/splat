"""Small immutable value objects shared across the domain and ports.

Pure Python + numpy only — no torch, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Licensing -----------------------------------------------------------


@dataclass(frozen=True)
class ModelLicense:
    """Provenance metadata for a model/pipeline. Surfaced loudly since the
    Gaussian Splatting model landscape mixes MIT/Apache with non-commercial
    and unconfirmed licenses.
    """

    spdx_id: str
    is_commercial: bool
    notes: str = ""

    def __str__(self) -> str:
        flag = "" if self.is_commercial else " (NON-COMMERCIAL)"
        return f"{self.spdx_id}{flag}"


MIT = ModelLicense(spdx_id="MIT", is_commercial=True)
BSD_3_CLAUSE = ModelLicense(spdx_id="BSD-3-Clause", is_commercial=True)
APACHE_2_0 = ModelLicense(spdx_id="Apache-2.0", is_commercial=True)
CC_BY_NC_SA_4_0 = ModelLicense(
    spdx_id="CC-BY-NC-SA-4.0",
    is_commercial=False,
    notes="Non-commercial use only.",
)
UNCONFIRMED = ModelLicense(
    spdx_id="UNCONFIRMED",
    is_commercial=False,
    notes="License could not be confirmed from public sources — treat as research-only.",
)
OPENRAIL_M = ModelLicense(
    spdx_id="OpenRAIL-M",
    is_commercial=True,
    notes="Commercial use generally permitted; carries use-based behavioral restrictions.",
)
SAI_NC_COMMUNITY = ModelLicense(
    spdx_id="StabilityAI-NC-Community",
    is_commercial=False,
    notes="Stability AI Non-Commercial Research Community License.",
)
APPLE_ASCL = ModelLicense(
    spdx_id="Apple-ASCL",
    is_commercial=True,
    notes="Apple Sample Code License — review specific terms before commercial redistribution.",
)


# --- Camera / pose ---------------------------------------------------------


@dataclass(frozen=True)
class Camera:
    """Pinhole camera intrinsics, in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class Pose:
    """Camera-to-world rigid transform."""

    position: np.ndarray  # (3,) float32
    rotation: np.ndarray  # (4,) float32 quaternion (w, x, y, z)

    def __post_init__(self) -> None:
        if self.position.shape != (3,):
            raise ValueError(f"Pose.position must have shape (3,), got {self.position.shape}")
        if self.rotation.shape != (4,):
            raise ValueError(f"Pose.rotation must have shape (4,), got {self.rotation.shape}")


# --- Coordinate conventions -------------------------------------------------

CoordinateConvention = str  # e.g. "colmap", "opengl", "gltf"

# Axis-flip matrices between the conventions this project actually needs to
# bridge. Kept as a small explicit table rather than a general transform
# graph — add a row only when a real adapter needs it.
_CONVENTION_FLIPS: dict[tuple[CoordinateConvention, CoordinateConvention], np.ndarray] = {
    ("colmap", "opengl"): np.diag([1.0, -1.0, -1.0]).astype(np.float32),
    ("opengl", "colmap"): np.diag([1.0, -1.0, -1.0]).astype(np.float32),
}


def convention_flip_matrix(
    source: CoordinateConvention, target: CoordinateConvention
) -> np.ndarray:
    """A 3x3 matrix mapping points/vectors from `source` into `target`."""
    if source == target:
        return np.eye(3, dtype=np.float32)
    try:
        return _CONVENTION_FLIPS[(source, target)]
    except KeyError as exc:
        raise ValueError(f"No known coordinate conversion from {source!r} to {target!r}") from exc
