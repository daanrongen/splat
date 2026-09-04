import pytest

from splat.domain.errors import SplatDomainError
from splat.handlers.gaussian import GaussianRequest, handle


class FakeReconstructionBackend:
    name = "fake-recon"

    def __init__(self, cloud) -> None:
        self._cloud = cloud

    def reconstruct(self, images, *, device="auto", **params):
        return self._cloud

    def required_image_count(self) -> tuple[int, int | None]:
        return (2, None)


def test_handle_writes_output_and_returns_cloud(mocker, tmp_path, synthetic_cloud):
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    output_path = tmp_path / "out.ply"
    request = GaussianRequest(
        inputs=[tmp_path / "a.png", tmp_path / "b.png"], output_path=output_path
    )

    result = handle(request)

    assert output_path.exists()
    assert result.cloud.point_count == synthetic_cloud.point_count


def test_handle_rejects_too_few_images(mocker, tmp_path, synthetic_cloud):
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )
    request = GaussianRequest(inputs=[tmp_path / "a.png"], output_path=tmp_path / "out.ply")

    with pytest.raises(SplatDomainError, match="requires between"):
        handle(request)
