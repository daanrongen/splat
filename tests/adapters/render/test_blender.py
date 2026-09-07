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
    _resolve_blender_bin,
)
from splat.domain.errors import RenderBackendError


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

    def fake_run(command, **kwargs):
        captured["command"] = command
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_bytes(b"fake png")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/blender")

    output = tmp_path / "render.png"
    BlenderBackend().render(synthetic_cloud, output, width=640, height=360, samples=8)

    assert output.read_bytes() == b"fake png"
    command = captured["command"]
    assert command[0] == "/usr/bin/blender"
    assert "--background" in command
    assert command[command.index("--width") + 1] == "640"
    assert command[command.index("--height") + 1] == "360"
    assert command[command.index("--samples") + 1] == "8"


def test_render_command_includes_engine_default_cycles(monkeypatch, tmp_path, synthetic_cloud):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        Path(command[command.index("--output") + 1]).write_bytes(b"fake png")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/blender")

    BlenderBackend().render(synthetic_cloud, tmp_path / "render.png")
    command = captured["command"]
    assert command[command.index("--engine") + 1] == "cycles"


def test_render_command_passes_through_custom_engine(monkeypatch, tmp_path, synthetic_cloud):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        Path(command[command.index("--output") + 1]).write_bytes(b"fake png")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/blender")

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
    np.testing.assert_allclose(data["rotations"], _convert_gaussian_rotations(cloud.rotations))
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


def test_render_raises_on_nonzero_exit(monkeypatch, tmp_path, synthetic_cloud):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/blender")

    with pytest.raises(RenderBackendError, match="boom"):
        BlenderBackend().render(synthetic_cloud, tmp_path / "render.png")
