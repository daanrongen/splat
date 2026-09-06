import subprocess
from pathlib import Path

import pytest

from splat.adapters.render.blender import BlenderBackend, _resolve_blender_bin
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


def test_render_raises_on_nonzero_exit(monkeypatch, tmp_path, synthetic_cloud):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/blender")

    with pytest.raises(RenderBackendError, match="boom"):
        BlenderBackend().render(synthetic_cloud, tmp_path / "render.png")
