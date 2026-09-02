from pathlib import Path

from typer.testing import CliRunner

from splat.adapters.formats.ply import PlyWriter
from splat.cli.main import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "convert" in result.output


def test_convert_ply_to_splat(tmp_path: Path, synthetic_cloud) -> None:
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    out_path = tmp_path / "out.splat"

    result = runner.invoke(app, ["convert", str(ply_path), str(out_path)])

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

    result = runner.invoke(app, ["compress", str(ply_path), str(out_path), "--profile", "archival"])

    assert result.exit_code == 0, result.output
    assert out_path.exists()


def test_export_stub_reports_not_implemented(tmp_path: Path) -> None:
    result = runner.invoke(app, ["export", str(tmp_path / "a.ply"), str(tmp_path / "b.obj")])
    assert result.exit_code == 1
    assert "not yet implemented" in result.output


def test_train_stub_reports_not_implemented(tmp_path: Path) -> None:
    result = runner.invoke(app, ["train", str(tmp_path)])
    assert result.exit_code == 1
    assert "not yet implemented" in result.output


def test_models_info() -> None:
    result = runner.invoke(app, ["models", "info", "mvsplat"])
    assert result.exit_code == 0, result.output
    assert "MIT" in result.output
