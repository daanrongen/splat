import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import RasterMetadata
from splat.registry.wiring import get_manifest_repository


def _put_image(monkeypatch, tmp_path, manifest_id: str, created_by: str = "diffuse:sdxl") -> None:
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    get_manifest_repository().put(
        manifest_id,
        kind=ManifestKind.IMAGE,
        content_bytes=manifest_id.encode(),
        ext="png",
        metadata=RasterMetadata(),
        parent_ids=[],
        created_by=created_by,
    )


def test_manifest_list_reports_cached_manifests(tmp_path, monkeypatch, call_tool):
    _put_image(monkeypatch, tmp_path, "a")

    result = call_tool("manifest_list")

    assert result.is_error is False
    rows = result.structured_content["result"]
    assert [row["id"] for row in rows] == ["a"]


def test_manifest_get_returns_full_detail(tmp_path, monkeypatch, call_tool):
    _put_image(monkeypatch, tmp_path, "a")

    result = call_tool("manifest_get", manifest_id="a")

    assert result.is_error is False
    body = json.loads(result.content[0].text)
    assert body["id"] == "a"
    assert body["created_by"] == "diffuse:sdxl"


def test_manifest_get_unknown_id_is_reported_as_tool_error(tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(ToolError):
        call_tool("manifest_get", manifest_id="does-not-exist")


def test_manifest_delete_removes_it(tmp_path, monkeypatch, call_tool):
    _put_image(monkeypatch, tmp_path, "a")

    result = call_tool("manifest_delete", manifest_id="a")

    assert result.is_error is False
    assert get_manifest_repository().find("a") is None
