import json
from pathlib import Path

from typer.testing import CliRunner

from splat.adapters.cache.filesystem import FilesystemAssetCache
from splat.cli.main import app
from splat.domain.asset import AssetKind
from splat.domain.value_objects import MIT
from tests.image_helpers import write_sample_png

runner = CliRunner()


class FakeReconstructionBackend:
    name = "fake-recon"
    license = MIT

    def __init__(self, cloud) -> None:
        self._cloud = cloud

    def reconstruct(self, images, *, device="auto", **params):
        return self._cloud

    def required_image_count(self) -> tuple[int, int | None]:
        return (2, None)


def _sample_image(tmp_path: Path, name: str) -> Path:
    return write_sample_png(tmp_path / name, (2, 2))


def _patch_backend(mocker, synthetic_cloud):
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )


def test_gaussian_file_inputs_emit_ndjson_asset(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker, synthetic_cloud)

    result = runner.invoke(
        app,
        [
            "gaussian",
            str(_sample_image(tmp_path, "a.png")),
            str(_sample_image(tmp_path, "b.png")),
            "--model",
            "fake-recon",
        ],
    )

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["kind"] == "gaussian_cloud"
    assert line["metadata"]["point_count"] == synthetic_cloud.point_count
    assert line["created_by"] == "gaussian:fake-recon"


def test_gaussian_asset_inputs_output_flag_writes_file(
    mocker, tmp_path, monkeypatch, synthetic_cloud
):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker, synthetic_cloud)
    cache = FilesystemAssetCache(tmp_path / "cache")
    a = cache.put_external(_sample_image(tmp_path, "a.png"), kind=AssetKind.IMAGE)
    b = cache.put_external(_sample_image(tmp_path, "b.png"), kind=AssetKind.IMAGE)
    out_path = tmp_path / "out.splat"

    result = runner.invoke(
        app,
        ["gaussian", f"@{a.id}", f"@{b.id}", "--model", "fake-recon", "-o", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["parent_ids"] == [a.id, b.id]


def test_gaussian_stdin_ndjson_input(mocker, tmp_path, monkeypatch, synthetic_cloud):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    _patch_backend(mocker, synthetic_cloud)
    cache = FilesystemAssetCache(tmp_path / "cache")
    a = cache.put_external(_sample_image(tmp_path, "a.png"), kind=AssetKind.IMAGE)
    b = cache.put_external(_sample_image(tmp_path, "b.png"), kind=AssetKind.IMAGE)
    stdin_payload = json.dumps({"id": a.id}) + "\n" + json.dumps({"id": b.id}) + "\n"

    result = runner.invoke(app, ["gaussian", "-", "--model", "fake-recon"], input=stdin_payload)

    assert result.exit_code == 0, result.output
    line = json.loads(result.output.strip().splitlines()[-1])
    assert line["parent_ids"] == [a.id, b.id]
