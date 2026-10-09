from fastapi.testclient import TestClient

from splat.http import gaussian as gaussian_route
from splat.http.app import app
from tests.image_helpers import sample_png_bytes

client = TestClient(app, raise_server_exceptions=False)


def test_unhandled_error_returns_500_with_detail(mocker):
    mocker.patch.object(gaussian_route, "handle", side_effect=RuntimeError("boom"))

    response = client.post("/gaussian", files={"images": ("a.png", sample_png_bytes())})

    assert response.status_code == 500
    assert response.json() == {"detail": "RuntimeError: boom"}
