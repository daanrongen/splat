from fastapi.testclient import TestClient

from splat.adapters.formats.ply import PlyWriter
from splat.http.app import app
from tests.adapters.mesh.test_poisson import _sphere_cloud

client = TestClient(app)


def test_extract_surface_returns_mesh_bytes(tmp_path):
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(_sphere_cloud(), ply_path)

    with ply_path.open("rb") as f:
        response = client.post(
            "/extract-surface",
            files={"input": ("in.ply", f, "application/octet-stream")},
            data={"to": "obj", "depth": 6},
        )

    assert response.status_code == 200, response.text
    assert len(response.content) > 0
    assert int(response.headers["X-Splat-Vertex-Count"]) > 0
    assert int(response.headers["X-Splat-Face-Count"]) > 0
