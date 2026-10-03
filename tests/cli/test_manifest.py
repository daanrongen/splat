import subprocess
import sys

from typer.testing import CliRunner

from splat.cli.main import app
from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import RasterMetadata
from splat.registry.wiring import get_manifest_repository
from tests.image_helpers import tiny_png

runner = CliRunner()


def _put_image(monkeypatch, tmp_path, manifest_id: str, created_by: str = "diffuse:sdxl") -> None:
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    get_manifest_repository().put(
        manifest_id,
        kind=ManifestKind.IMAGE,
        content_bytes=tiny_png(manifest_id),
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=[],
        created_by=created_by,
    )


def test_manifest_list_shows_cached_manifests(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "abc123")

    result = runner.invoke(app, ["manifest", "list"])

    assert result.exit_code == 0
    assert "abc123" in result.stdout
    assert "diffuse:sdxl" in result.stdout


def test_manifest_get_shows_full_detail(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "abc123")

    result = runner.invoke(app, ["manifest", "get", "abc123"])

    assert result.exit_code == 0
    assert "abc123" in result.stdout
    assert "diffuse:sdxl" in result.stdout


def test_manifest_get_unknown_id_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))

    result = runner.invoke(app, ["manifest", "get", "does-not-exist"])

    assert result.exit_code == 1


def test_manifest_rm_deletes_the_manifest(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "abc123")

    result = runner.invoke(app, ["manifest", "rm", "abc123"])

    assert result.exit_code == 0
    assert get_manifest_repository().find("abc123") is None


def test_manifest_clear_with_yes_deletes_without_prompting(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "abc123")
    _put_image(monkeypatch, tmp_path, "def456")

    result = runner.invoke(app, ["manifest", "clear", "--yes"])

    assert result.exit_code == 0, result.output
    assert "cleared 2 manifest(s)" in result.output
    assert get_manifest_repository().list() == []


def test_manifest_clear_prompts_and_aborts_on_no(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "abc123")

    result = runner.invoke(app, ["manifest", "clear"], input="n\n")

    assert result.exit_code != 0
    assert get_manifest_repository().find("abc123") is not None


def test_manifest_clear_prompts_and_deletes_on_yes(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "abc123")

    result = runner.invoke(app, ["manifest", "clear"], input="y\n")

    assert result.exit_code == 0, result.output
    assert get_manifest_repository().find("abc123") is None


def test_manifest_clear_with_no_matches_does_not_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))

    result = runner.invoke(app, ["manifest", "clear"])

    assert result.exit_code == 0, result.output
    assert "nothing to clear" in result.output


def test_manifest_clear_respects_created_by_filter(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "abc123", created_by="diffuse:sdxl")
    _put_image(monkeypatch, tmp_path, "def456", created_by="upscale:realesrgan")

    result = runner.invoke(app, ["manifest", "clear", "--created-by", "diffuse", "--yes"])

    assert result.exit_code == 0, result.output
    assert get_manifest_repository().find("abc123") is None
    assert get_manifest_repository().find("def456") is not None


def test_manifest_command_path_does_not_import_heavy_backends():
    code = """
import click
import sys
import typer
import splat.cli.main

cmd = typer.main.get_command(splat.cli.main.app)
cmd.get_command(click.Context(cmd), "manifest")
for name in ("torch", "coremltools", "mlx", "transformers", "trimesh", "cv2"):
    assert name not in sys.modules, name
assert "splat.application.pipeline" not in sys.modules
assert [m for m in sys.modules if m.startswith("splat.handlers.")] == ["splat.handlers.manifest"]
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_manifest_export_writes_file_and_sidecar(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "abc123")

    result = runner.invoke(app, ["manifest", "export", "abc123", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "abc123.png").read_bytes() == tiny_png("abc123")
    assert (tmp_path / "out" / "abc123.png.manifest.json").exists()


def test_manifest_export_unknown_id_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))

    result = runner.invoke(app, ["manifest", "export", "nope", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert not (tmp_path / "out").exists()


def test_manifest_label_passes_piped_records_through(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "abc123")

    result = runner.invoke(app, ["manifest", "label", "-", "robot"], input='{"id": "abc123"}\n')

    assert result.exit_code == 0, result.output
    assert '"id": "abc123"' in result.stdout
    assert get_manifest_repository().get("abc123").label == "robot"


def test_manifest_get_prints_the_lineage_tree(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "parent1")
    get_manifest_repository().put(
        "child1",
        kind=ManifestKind.IMAGE,
        content_bytes=tiny_png("child"),
        ext="png",
        metadata=RasterMetadata(),
        params={"factor": 2},
        parent_ids=["parent1"],
        created_by="upscale:realesrgan-mlx",
    )

    result = runner.invoke(app, ["manifest", "get", "child1"])

    assert result.exit_code == 0, result.output
    assert "parent1" in result.stdout
    assert "BSD-3-Clause" in result.stdout


def test_manifest_rm_refuses_without_cascade(tmp_path, monkeypatch):
    _put_image(monkeypatch, tmp_path, "parent1")
    get_manifest_repository().put(
        "child1",
        kind=ManifestKind.IMAGE,
        content_bytes=tiny_png("child"),
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=["parent1"],
        created_by="test",
    )

    refused = runner.invoke(app, ["manifest", "rm", "parent1"])
    cascaded = runner.invoke(app, ["manifest", "rm", "parent1", "--cascade"])

    assert refused.exit_code == 1
    assert cascaded.exit_code == 0
    assert get_manifest_repository().list() == []
