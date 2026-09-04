from io import StringIO
from pathlib import Path

from typer.testing import CliRunner

from splat.adapters.cache.filesystem import FilesystemAssetCache
from splat.adapters.formats.ply import PlyWriter
from splat.cli._pipeline_io import resolve_inputs
from splat.cli.main import app
from splat.domain.manifest import ManifestKind
from tests.image_helpers import write_sample_png

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "tools" in result.output
    assert "caption" in result.output
    assert "upscale" in result.output


def test_piped_asset_resolution_ignores_non_json_chatter(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    cache = FilesystemAssetCache(tmp_path / "cache")
    asset = cache.put_external(
        write_sample_png(tmp_path / "scene.png", (2, 2)), kind=ManifestKind.IMAGE
    )
    monkeypatch.setattr(
        "sys.stdin",
        StringIO(f'Torch warning emitted by imported dependency\n{{"id": "{asset.id}"}}\n'),
    )

    assert resolve_inputs("-", cache) == [asset]


def test_convert_and_compress_are_not_top_level_commands() -> None:
    convert = runner.invoke(app, ["convert", "--help"])
    compress = runner.invoke(app, ["compress", "--help"])

    assert convert.exit_code == 2
    assert compress.exit_code == 2


def test_tools_help_lists_deterministic_transforms() -> None:
    result = runner.invoke(app, ["tools", "--help"])

    assert result.exit_code == 0
    assert "convert" in result.output
    assert "compress" in result.output
    assert "displace.height" in result.output


def test_convert_ply_to_splat(tmp_path: Path, synthetic_cloud) -> None:
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    out_path = tmp_path / "out.splat"

    result = runner.invoke(app, ["tools", "convert", str(ply_path), str(out_path)])

    assert result.exit_code == 0, result.output
    assert out_path.exists()


def test_info(tmp_path: Path, synthetic_cloud) -> None:
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    result = runner.invoke(app, ["info", str(ply_path)])

    assert result.exit_code == 0, result.output
    assert "points:" in result.output


def test_validate(tmp_path: Path, synthetic_cloud) -> None:
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    result = runner.invoke(app, ["validate", str(ply_path), "--strict"])

    assert result.exit_code == 0, result.output


def test_compress(tmp_path: Path, synthetic_cloud) -> None:
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    out_path = tmp_path / "out.ply"

    result = runner.invoke(
        app, ["tools", "compress", str(ply_path), str(out_path), "--profile", "archival"]
    )

    assert result.exit_code == 0, result.output
    assert out_path.exists()


def test_gaussian_unknown_model_errors(tmp_path: Path) -> None:
    a = write_sample_png(tmp_path / "a.png", (2, 2))
    b = write_sample_png(tmp_path / "b.png", (2, 2))

    result = runner.invoke(
        app,
        [
            "gaussian",
            str(a),
            str(b),
            "-o",
            str(tmp_path / "out.splat"),
            "--model",
            "not-a-real-model",
        ],
    )
    assert result.exit_code == 1
    assert "Unknown model" in result.output


def test_train_stub_reports_not_implemented(tmp_path: Path) -> None:
    result = runner.invoke(app, ["train", str(tmp_path)])
    assert result.exit_code == 1
    assert "not yet implemented" in result.output


def test_train_help_has_no_backend_options() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--backend" not in result.output
    assert "--iterations" not in result.output


def test_models_info() -> None:
    result = runner.invoke(app, ["models", "info", "mvsplat"])
    assert result.exit_code == 0, result.output
    assert "MIT" in result.output
