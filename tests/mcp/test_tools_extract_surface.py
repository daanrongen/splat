from splat.adapters.formats.ply import PlyWriter
from tests.adapters.mesh.test_poisson import _sphere_cloud


def test_extract_surface_writes_a_mesh(tmp_path, monkeypatch, call_tool):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(_sphere_cloud(), ply_path)
    out_path = tmp_path / "out.obj"

    result = call_tool(
        "tools_extract_surface",
        input_path=str(ply_path),
        output_path=str(out_path),
        depth=6,
    )

    assert result.is_error is False
    assert out_path.exists()
