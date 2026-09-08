import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from splat.adapters.render.blender import (
    BlenderBackend,
    _cloud_to_npz,
    _convert_camera_rotation,
    _convert_gaussian_rotations,
    _convert_positions,
    _gaussian_axes,
    _parse_look_at,
    _resolve_blender_bin,
    _resolve_timeout,
    _srgb_to_linear,
)
from splat.domain.errors import RenderBackendError


class FakePopen:
    """Minimal Popen stand-in: the adapter iterates stdout, then waits."""

    def __init__(self, lines: list[str], returncode: int = 0, wait_raises=None) -> None:
        self.stdout = iter(f"{line}\n" for line in lines)
        self._returncode = returncode
        self._wait_raises = wait_raises
        self.killed = False

    def wait(self, timeout=None):
        if self._wait_raises is not None:
            raise self._wait_raises
        return self._returncode

    def kill(self):
        # A real process stops timing out once killed; the adapter waits again
        # after killing, and this fake has to stop raising or that wait escapes.
        self.killed = True
        self._wait_raises = None


def _patch_popen(monkeypatch, captured, *, lines=(), returncode=0, wait_raises=None):
    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output = Path(command[command.index("--output") + 1])
        if returncode == 0 and wait_raises is None:
            output.write_bytes(b"fake png")
        return FakePopen(list(lines), returncode, wait_raises)

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/blender")


def test_resolve_blender_bin_raises_when_not_found(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.delenv("SPLAT_BLENDER_BIN", raising=False)
    with pytest.raises(RenderBackendError):
        _resolve_blender_bin(None)


def test_resolve_blender_bin_prefers_explicit_over_env_and_path(monkeypatch):
    monkeypatch.setenv("SPLAT_BLENDER_BIN", "/env/blender")
    monkeypatch.setattr("shutil.which", lambda name: "/path/blender")
    assert _resolve_blender_bin("/explicit/blender") == "/explicit/blender"


def test_resolve_blender_bin_falls_back_to_env_then_path(monkeypatch):
    monkeypatch.delenv("SPLAT_BLENDER_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/path/blender")
    assert _resolve_blender_bin(None) == "/path/blender"


def test_render_invokes_blender_with_expected_command(monkeypatch, tmp_path, synthetic_cloud):
    captured = {}

    _patch_popen(monkeypatch, captured)

    output = tmp_path / "render.png"
    BlenderBackend().render(synthetic_cloud, output, width=640, height=360, samples=8)

    assert output.read_bytes() == b"fake png"
    command = captured["command"]
    assert command[0] == "/usr/bin/blender"
    assert "--background" in command  # Blender's own headless flag
    assert command[command.index("--width") + 1] == "640"
    assert command[command.index("--height") + 1] == "360"
    assert command[command.index("--samples") + 1] == "8"


def test_render_command_includes_engine_default_cycles(monkeypatch, tmp_path, synthetic_cloud):
    captured = {}

    _patch_popen(monkeypatch, captured)

    BlenderBackend().render(synthetic_cloud, tmp_path / "render.png")
    command = captured["command"]
    assert command[command.index("--engine") + 1] == "cycles"


def test_render_command_passes_through_custom_engine(monkeypatch, tmp_path, synthetic_cloud):
    captured = {}

    _patch_popen(monkeypatch, captured)

    BlenderBackend().render(synthetic_cloud, tmp_path / "render.png", engine="eevee")
    command = captured["command"]
    assert command[command.index("--engine") + 1] == "eevee"


def test_convert_positions_flips_y_and_z():
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(_convert_positions(points), [[1.0, -2.0, -3.0]])


def test_convert_gaussian_rotations_negates_y_and_z_components():
    quats = np.array([[0.7071, 0.0, 0.7071, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(
        _convert_gaussian_rotations(quats), [[0.7071, 0.0, -0.7071, 0.0]], atol=1e-6
    )


def test_convert_camera_rotation_matches_manual_matrix_conjugation():
    # an arbitrary orthonormal world-to-camera rotation (90deg about Z)
    r_world_to_cam = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    flip = np.diag([1.0, -1.0, -1.0])
    expected = flip @ np.asarray(r_world_to_cam).T @ flip
    np.testing.assert_allclose(_convert_camera_rotation(r_world_to_cam), expected, atol=1e-6)


def test_cloud_to_npz_converts_colmap_cloud_and_carries_camera_pose(tmp_path, synthetic_cloud):
    cloud = replace(
        synthetic_cloud,
        metadata=replace(
            synthetic_cloud.metadata,
            coordinate_convention="colmap",
            capture_camera_position=[1.0, 2.0, 3.0],
            capture_camera_rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            capture_camera_intrinsics=[800.0, 800.0, 320.0, 240.0, 640.0, 480.0],
        ),
    )
    out = tmp_path / "cloud.npz"
    _cloud_to_npz(cloud, out)
    data = np.load(out)

    np.testing.assert_allclose(data["means"], _convert_positions(cloud.means))
    np.testing.assert_allclose(
        data["axes"],
        _gaussian_axes(_convert_gaussian_rotations(cloud.rotations), cloud.to_linear_scales()),
    )
    np.testing.assert_allclose(
        data["camera_position"], _convert_positions(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    )
    assert "camera_rotation" in data
    assert "camera_intrinsics" in data


def test_cloud_to_npz_skips_axis_conversion_for_non_colmap_convention(tmp_path, synthetic_cloud):
    cloud = replace(
        synthetic_cloud, metadata=replace(synthetic_cloud.metadata, coordinate_convention="opengl")
    )
    out = tmp_path / "cloud.npz"
    _cloud_to_npz(cloud, out)
    data = np.load(out)
    np.testing.assert_allclose(data["means"], cloud.means)
    assert "camera_position" not in data


def test_render_command_passes_the_background_through(monkeypatch, tmp_path, synthetic_cloud):
    captured = {}

    _patch_popen(monkeypatch, captured)

    BlenderBackend().render(synthetic_cloud, tmp_path / "render.png", background="transparent")
    command = captured["command"]
    assert command[command.index("--background-color") + 1] == "transparent"


def test_render_omits_camera_flags_that_were_not_asked_for(monkeypatch, tmp_path, synthetic_cloud):
    """The script needs to tell "no viewpoint given" (use the capture pose)
    from an explicit one, so a defaulted value must not appear in argv."""
    captured = {}

    _patch_popen(monkeypatch, captured)

    BlenderBackend().render(synthetic_cloud, tmp_path / "render.png", azimuth=90.0)
    command = captured["command"]
    assert command[command.index("--azimuth") + 1] == "90.0"
    assert "--elevation" not in command
    assert "--distance" not in command
    assert "--look-at" not in command


def test_render_normalizes_look_at(monkeypatch, tmp_path, synthetic_cloud):
    captured = {}

    _patch_popen(monkeypatch, captured)

    BlenderBackend().render(synthetic_cloud, tmp_path / "render.png", look_at=" 1, 2 ,3 ")
    command = captured["command"]
    assert command[command.index("--look-at") + 1] == "1.0,2.0,3.0"


@pytest.mark.parametrize("spec", ["1,2", "1,2,3,4", "a,b,c", ""])
def test_parse_look_at_rejects_anything_but_three_numbers(spec):
    with pytest.raises(RenderBackendError, match="look-at"):
        _parse_look_at(spec)


def test_gaussian_axes_of_an_unrotated_kernel_are_the_inverse_scales():
    quats = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    scales = np.array([[0.5, 0.25, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(
        _gaussian_axes(quats, scales)[0], np.diag([2.0, 4.0, 0.5]), atol=1e-6
    )


def test_gaussian_axes_map_a_one_sigma_offset_to_unit_length():
    """The point of the basis: `axes @ x` is the Mahalanobis coordinate, so an
    offset of exactly one sigma along a principal axis comes back as 1.0."""
    angle = np.pi / 3
    quats = np.array([[np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]], dtype=np.float32)
    scales = np.array([[0.3, 0.1, 0.7]], dtype=np.float32)
    axes = _gaussian_axes(quats, scales)[0]

    # the rotated first principal axis, one sigma out
    offset = 0.3 * np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
    np.testing.assert_allclose(axes @ offset, [1.0, 0.0, 0.0], atol=1e-6)


def test_gaussian_axes_tolerate_a_degenerate_zero_scale():
    quats = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    scales = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    assert np.isfinite(_gaussian_axes(quats, scales)).all()


def test_srgb_to_linear_matches_the_standard_at_both_ends_and_mid_grey():
    values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    np.testing.assert_allclose(_srgb_to_linear(values), [0.0, 0.21404, 1.0], atol=1e-5)


def test_cloud_to_npz_stores_linearized_colours(tmp_path, synthetic_cloud):
    """A washed-out frame was the symptom: SH colours are sRGB display values,
    and Blender's Standard view transform re-encodes whatever it is handed."""
    out = tmp_path / "cloud.npz"
    _cloud_to_npz(synthetic_cloud, out)
    colors = np.load(out)["colors"]
    display = np.clip(0.5 + 0.28209479177387814 * synthetic_cloud.sh_dc, 0.0, 1.0)
    np.testing.assert_allclose(colors, _srgb_to_linear(display), atol=1e-6)


def test_cloud_to_npz_sizes_billboards_to_three_sigma(tmp_path, synthetic_cloud):
    out = tmp_path / "cloud.npz"
    _cloud_to_npz(synthetic_cloud, out)
    data = np.load(out)
    np.testing.assert_allclose(
        data["radii"], 3.0 * synthetic_cloud.to_linear_scales().max(axis=1), rtol=1e-6
    )


def test_render_raises_on_nonzero_exit(monkeypatch, tmp_path, synthetic_cloud):
    _patch_popen(monkeypatch, {}, lines=["Error: boom"], returncode=1)

    with pytest.raises(RenderBackendError, match="boom"):
        BlenderBackend().render(synthetic_cloud, tmp_path / "render.png")


def test_resolve_timeout_defaults_to_thirty_minutes(monkeypatch):
    monkeypatch.delenv("SPLAT_RENDER_TIMEOUT", raising=False)
    assert _resolve_timeout() == 1800.0


def test_resolve_timeout_env_zero_disables_it(monkeypatch):
    monkeypatch.setenv("SPLAT_RENDER_TIMEOUT", "0")
    assert _resolve_timeout() is None


def test_render_translates_timeout_to_domain_error(monkeypatch, tmp_path, synthetic_cloud):
    monkeypatch.setenv("SPLAT_RENDER_TIMEOUT", "5")
    _patch_popen(
        monkeypatch,
        {},
        wait_raises=subprocess.TimeoutExpired(cmd="blender", timeout=5),
    )

    with pytest.raises(RenderBackendError, match="exceeded 5s"):
        BlenderBackend().render(synthetic_cloud, tmp_path / "render.png")


def test_render_forwards_reported_progress_and_not_blenders_noise(
    monkeypatch, tmp_path, synthetic_cloud
):
    """The whole point of streaming: Blender is silent in background mode, so
    only the lines `_blender_script.py` reports are useful to a caller."""
    _patch_popen(
        monkeypatch,
        {},
        lines=[
            "Blender 5.2.1 LTS",
            "splat| device Apple M1 Pro (GPU - 16 cores) [METAL]",
            "splat| Mem: 3143M | Sample 12/64",
            "Saved: '/tmp/x.png'",
        ],
    )
    seen: list[str] = []

    BlenderBackend().render(synthetic_cloud, tmp_path / "render.png", on_progress=seen.append)

    assert seen == [
        "device Apple M1 Pro (GPU - 16 cores) [METAL]",
        "Mem: 3143M | Sample 12/64",
    ]


def test_render_works_without_a_progress_callback(monkeypatch, tmp_path, synthetic_cloud):
    _patch_popen(monkeypatch, {}, lines=["splat| Sample 1/8"])

    BlenderBackend().render(synthetic_cloud, tmp_path / "render.png")

    assert (tmp_path / "render.png").read_bytes() == b"fake png"
