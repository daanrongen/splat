import json
import subprocess
import sys

import pytest

HEAVY = [
    "torch",
    "transformers",
    "huggingface_hub",
    "mlx",
    "coremltools",
    "open3d",
    "cv2",
    "scipy",
    "trimesh",
]

PROBE = """
import json, sys
from splat.cli.main import app

sys.argv = ["splat", *{args!r}]
try:
    app()
except SystemExit:
    pass
print(json.dumps(sorted({heavy!r} & {{m.split(".")[0] for m in sys.modules}})))
"""


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["env"],
        ["manifest", "list"],
        ["models", "list"],
        ["gaussian", "--help"],
        ["mesh", "--help"],
        ["export", "--help"],
    ],
    ids=" ".join,
)
def test_light_commands_import_no_heavy_runtime(args, tmp_path):
    env = {
        "PATH": "",
        "SPLAT_MANIFEST_CACHE_DIR": str(tmp_path / "manifests"),
        "SPLAT_MODEL_CACHE_DIR": str(tmp_path / "models"),
        "HF_HOME": str(tmp_path / "hf"),
    }
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(args=args, heavy=set(HEAVY))],
        capture_output=True,
        text=True,
        env=env,
    )

    assert json.loads(result.stdout.splitlines()[-1]) == [], result.stderr
