from fastapi.testclient import TestClient

from splat.adapters.formats.ply import PlyWriter
from splat.http.app import app

client = TestClient(app)


def test_declutter_returns_output_bytes(tmp_path, synthetic_cloud):
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    with ply_path.open("rb") as f:
        response = client.post(
            "/declutter", files={"input": ("in.ply", f, "application/octet-stream")}
        )

    assert response.status_code == 200, response.text
    assert len(response.content) > 0
    assert "X-Splat-Point-Count" in response.headers
