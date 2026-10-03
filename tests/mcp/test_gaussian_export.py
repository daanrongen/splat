from splat.adapters.formats.ply import PlyWriter
from splat.domain.value_objects import MIT
from tests.image_helpers import write_sample_png


class FakeReconstructionBackend:
    name = "fake-recon"
    license = MIT
    provides_metric_scale = False

    def __init__(self, cloud) -> None:
        self._cloud = cloud

    def reconstruct(self, images, *, device="auto", **params):
        return self._cloud

    def required_image_count(self) -> tuple[int, int | None]:
        return (2, None)


def test_gaussian_writes_output_file(mocker, tmp_path, synthetic_cloud, call_tool):
    mocker.patch.dict("os.environ", {"SPLAT_MANIFEST_CACHE_DIR": str(tmp_path / "cache")})
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    output_path = tmp_path / "out.ply"
    a = write_sample_png(tmp_path / "a.png", (2, 2))
    b = write_sample_png(tmp_path / "b.png", (2, 2))

    result = call_tool(
        "gaussian",
        images=[str(a), str(b)],
        output_path=str(output_path),
        model="fake-recon",
    )

    assert result.is_error is False
    assert output_path.exists()


def test_export_writes_output_file(tmp_path, monkeypatch, synthetic_cloud, call_tool):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    splat_path = tmp_path / "out.splat"

    result = call_tool(
        "export", input=str(ply_path), output_path=str(splat_path), profile="web-delivery"
    )

    assert result.is_error is False
    assert splat_path.exists()
