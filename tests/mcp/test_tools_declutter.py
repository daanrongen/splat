from splat.adapters.formats.ply import PlyWriter


def test_declutter_writes_output_file(tmp_path, synthetic_cloud, call_tool):
    ply_path = tmp_path / "in.ply"
    PlyWriter().write(synthetic_cloud, ply_path)
    out_path = tmp_path / "out.ply"

    result = call_tool("tools_declutter", input_path=str(ply_path), output_path=str(out_path))

    assert result.is_error is False
    assert out_path.exists()
