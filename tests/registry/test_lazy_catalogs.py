import subprocess
import sys

_BLOCK_BACKEND_DEPS = """
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("mlx", "coremltools", "torch"):
            raise ImportError(name)
sys.meta_path.insert(0, Block())
import splat.mcp.server
"""


def test_mcp_server_imports_without_backend_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_BACKEND_DEPS], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
