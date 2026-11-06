from fastapi.testclient import TestClient

from splat.http.app import app

client = TestClient(app)


def test_open_without_a_token(monkeypatch):
    monkeypatch.delenv("SPLAT_TOKEN", raising=False)

    assert client.get("/models").status_code == 200


def test_requires_the_token(monkeypatch):
    monkeypatch.setenv("SPLAT_TOKEN", "s3cret")

    assert client.get("/models").status_code == 401
    assert client.get("/models", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_version_stays_open(monkeypatch):
    monkeypatch.setenv("SPLAT_TOKEN", "s3cret")

    assert client.get("/version").status_code == 200


def test_remote_client_sends_the_token(monkeypatch):
    from splat.adapters.client.http import RemoteSplatClient

    monkeypatch.setenv("SPLAT_TOKEN", "s3cret")

    remote = RemoteSplatClient("http://splat.test")

    assert remote._client.headers["Authorization"] == "Bearer s3cret"
