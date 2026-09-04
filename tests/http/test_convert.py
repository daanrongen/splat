from fastapi.testclient import TestClient

from splat.adapters.formats.ply import PlyWriter
from splat.http.app import app

client = TestClient(app)


def test_convert_ply_to_splat(tmp_path, synthetic_cloud):
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)

    with ply_path.open("rb") as f:
        response = client.post(
            "/convert",
            files={"input": ("in.ply", f, "application/octet-stream")},
            data={"to": "splat"},
        )

    assert response.status_code == 200, response.text
    assert len(response.content) > 0
    assert response.headers["X-Splat-Point-Count"] == str(synthetic_cloud.point_count)
