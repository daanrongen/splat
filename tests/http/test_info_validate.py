from fastapi.testclient import TestClient

from splat.adapters.formats.ply import PlyWriter
from splat.http.app import app

client = TestClient(app)


def test_info_reports_point_count(tmp_path, synthetic_cloud):
    ply_path = tmp_path / "a.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    with ply_path.open("rb") as f:
        response = client.post("/info", files={"input": ("a.ply", f, "application/octet-stream")})

    assert response.status_code == 200, response.text
    assert response.json()["points"] == synthetic_cloud.point_count


def test_validate_reports_valid(tmp_path, synthetic_cloud):
    ply_path = tmp_path / "a.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    with ply_path.open("rb") as f:
        response = client.post(
            "/validate", files={"input": ("a.ply", f, "application/octet-stream")}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["points"] == synthetic_cloud.point_count
    assert isinstance(body["issues"], list)
