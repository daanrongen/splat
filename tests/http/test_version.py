from fastapi.testclient import TestClient

from splat import __version__
from splat.http.app import app


def test_version_reports_the_package_version():
    response = TestClient(app).get("/version")

    assert response.json() == {"version": __version__}
