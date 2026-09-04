import io

from fastapi.testclient import TestClient
from PIL import Image

from splat.adapters.formats.ply import PlyWriter
from splat.http.app import app

client = TestClient(app)


class FakeReconstructionBackend:
    name = "fake-recon"

    def __init__(self, cloud) -> None:
        self._cloud = cloud

    def reconstruct(self, images, *, device="auto", **params):
        return self._cloud

    def required_image_count(self) -> tuple[int, int | None]:
        return (2, None)


def _sample_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="PNG")
    return buf.getvalue()


def test_gaussian_returns_output_bytes(mocker, synthetic_cloud):
    mocker.patch("splat.handlers.gaussian.get_model_source", return_value=object())
    mocker.patch(
        "splat.handlers.gaussian.get_reconstruction_backend",
        return_value=FakeReconstructionBackend(synthetic_cloud),
    )

    response = client.post(
        "/gaussian",
        files=[
            ("images", ("a.png", _sample_png_bytes(), "image/png")),
            ("images", ("b.png", _sample_png_bytes(), "image/png")),
        ],
        data={"to": "ply"},
    )

    assert response.status_code == 200, response.text
    assert len(response.content) > 0
    assert response.headers["X-Splat-Point-Count"] == str(synthetic_cloud.point_count)


def test_compress_returns_output_bytes(tmp_path, synthetic_cloud):
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    with ply_path.open("rb") as f:
        response = client.post(
            "/compress", files={"input": ("in.ply", f, "application/octet-stream")}
        )

    assert response.status_code == 200, response.text
    assert len(response.content) > 0
