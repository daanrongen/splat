from typer.testing import CliRunner

from splat.cli.main import app
from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import RasterMetadata
from splat.registry.wiring import get_manifest_repository

runner = CliRunner()


def _put_image(monkeypatch, tmp_path, manifest_id: str, created_by: str = "diffuse:sdxl") -> None:
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    get_manifest_repository().put(
        manifest_id,
        kind=ManifestKind.IMAGE,
        content_bytes=manifest_id.encode(),
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
