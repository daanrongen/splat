"""Opt-in: runs a real Blender render. Needs `blender` on PATH plus
SPLAT_INTEGRATION_TESTS=1, and takes a few seconds for the Metal kernels.

`_blender_script.py` had 0% coverage, which is how it shipped rendering
Gaussians as lit opaque spheres (#82). Every assertion here is on the image
formation model rather than on pixels: one Gaussian of a known colour,
opacity and sigma, at a known distance, has an analytically knowable alpha at
every pixel, and that is what is checked.
"""

import os
import shutil

import numpy as np
import pytest

from splat.adapters.render.blender import BlenderBackend
from splat.domain.gaussians import GaussianCloud, GaussianCloudMetadata

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("SPLAT_INTEGRATION_TESTS") != "1",
        reason="set SPLAT_INTEGRATION_TESTS=1 to run a real Blender render",
    ),
    pytest.mark.skipif(shutil.which("blender") is None, reason="blender not on PATH"),
]

_SH_C0 = 0.28209479177387814
_SIZE = 64
_SIGMA = 0.1
_OPACITY_LOGIT = 6.0
_DISTANCE = 1.0


@pytest.fixture
def one_red_gaussian() -> GaussianCloud:
    """Pure red (1, 0, 0) after SH activation, isotropic, at the origin, framed
    by a camera one unit away on +Z with a 64px focal length - so one pixel of
    image offset is exactly 1/64 of a world unit at the Gaussian's depth."""
    return GaussianCloud(
        means=np.zeros((1, 3), dtype=np.float32),
        scales=np.full((1, 3), np.log(_SIGMA), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([_OPACITY_LOGIT], dtype=np.float32),
        sh_dc=np.array([[0.5, -0.5, -0.5]], dtype=np.float32) / _SH_C0,
        metadata=GaussianCloudMetadata(
            coordinate_convention="opengl",
            capture_camera_position=[0.0, 0.0, _DISTANCE],
            capture_camera_rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            capture_camera_intrinsics=[
                float(_SIZE),
                float(_SIZE),
                _SIZE / 2,
                _SIZE / 2,
                float(_SIZE),
                float(_SIZE),
            ],
        ),
    )


def _render(cloud, tmp_path, **params) -> np.ndarray:
    from PIL import Image

    output = tmp_path / "render.png"
    params.setdefault("background", "black")
    BlenderBackend().render(cloud, output, width=_SIZE, height=_SIZE, samples=16, **params)
    return np.asarray(Image.open(output).convert("RGBA"), dtype=np.float32) / 255.0


def _expected_alpha(pixel_offset: int) -> float:
    """3DGS alpha for a ray `pixel_offset` pixels off centre: opacity times
    exp(-0.5 m^2), with m the closest approach of that ray to the Gaussian
    centre in sigmas. The ray is not perpendicular to the offset, hence the
    1/hypot term rather than a flat division."""
    offset = pixel_offset / _SIZE * _DISTANCE
    perpendicular = offset / np.hypot(1.0, offset / _DISTANCE)
    opacity = 1.0 / (1.0 + np.exp(-_OPACITY_LOGIT))
    return float(opacity * np.exp(-0.5 * (perpendicular / _SIGMA) ** 2))


def test_alpha_follows_the_gaussian_falloff(one_red_gaussian, tmp_path):
    pixels = _render(one_red_gaussian, tmp_path, background="transparent")
    centre = _SIZE // 2

    measured = [pixels[centre, centre + offset, 3] for offset in (0, 2, 4, 6, 8)]
    expected = [_expected_alpha(offset) for offset in (0, 2, 4, 6, 8)]

    # spans ~0.99 down to ~0.04, so this is the shape of the curve and not a
    # coincidence of tolerance; the slack absorbs Cycles' pixel filter.
    np.testing.assert_allclose(measured, expected, atol=0.12)


def test_kernel_support_ends_well_inside_the_frame(one_red_gaussian, tmp_path):
    """3 sigma is 19 pixels here, so the frame corners have to be untouched -
    a hard-edged proxy sphere or a missing falloff both show up as coverage."""
    pixels = _render(one_red_gaussian, tmp_path, background="transparent")

    assert pixels[0, 0, 3] == 0.0
    assert pixels[-1, -1, 3] == 0.0


@pytest.fixture
def axis_markers() -> GaussianCloud:
    """Red above, blue below, green on +X, no capture pose - so the orbit
    camera has to place them, and where they land says which axis it treated
    as up and which way azimuth turns."""
    return GaussianCloud(
        means=np.array([[0, 1, 0], [0, -1, 0], [1, 0, 0]], dtype=np.float32),
        scales=np.full((3, 3), np.log(0.15), dtype=np.float32),
        rotations=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)).astype(np.float32),
        opacities=np.full(3, _OPACITY_LOGIT, dtype=np.float32),
        sh_dc=np.array([[0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5]], dtype=np.float32)
        / _SH_C0,
        metadata=GaussianCloudMetadata(coordinate_convention="opengl"),
    )


def _brightest(pixels: np.ndarray, channel: int) -> tuple[int, int]:
    others = [index for index in range(3) if index != channel]
    score = pixels[:, :, channel] - pixels[:, :, others].max(axis=2)
    row, column = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(row), int(column)


def test_orbit_camera_treats_plus_y_as_up(axis_markers, tmp_path):
    """Blender's world up is +Z and clouds are handed over in the OpenGL
    convention, so anything framed with `to_track_quat`'s up hint comes out
    upside down (issue #83)."""
    pixels = _render(axis_markers, tmp_path, azimuth=0.0, elevation=0.0, distance=5.0)

    red_row, _ = _brightest(pixels, 0)
    blue_row, _ = _brightest(pixels, 2)

    assert red_row < _SIZE // 2 < blue_row


def test_orbit_azimuth_turns_toward_plus_x(axis_markers, tmp_path):
    front = _render(axis_markers, tmp_path, azimuth=0.0, elevation=0.0, distance=5.0)
    side = _render(axis_markers, tmp_path, azimuth=90.0, elevation=0.0, distance=5.0)

    # green sits on +X: to the right of centre head-on, and dead centre once
    # the camera has swung a quarter turn onto its axis.
    assert _brightest(front, 1)[1] > _SIZE // 2 + 4
    assert abs(_brightest(side, 1)[1] - _SIZE // 2) <= 2


def test_colour_is_the_gaussians_own_and_not_a_lighting_response(one_red_gaussian, tmp_path):
    """The old model shaded a Principled BSDF under a sun lamp, so this cloud
    rendered as whatever red does under that light. Emission means the pixel is
    the SH colour, and with no lights a BSDF would render black instead."""
    pixels = _render(one_red_gaussian, tmp_path, background="black")
    centre = pixels[_SIZE // 2, _SIZE // 2]

    np.testing.assert_allclose(centre[:3], [1.0, 0.0, 0.0], atol=0.02)
