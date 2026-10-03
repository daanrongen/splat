"""One real run per catalogued model, asserting its output kind's contract.
Opt in with SPLAT_INTEGRATION_TESTS=1; downloads weights on first use."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import splat
from splat.domain.manifest import ManifestKind
from splat.domain.manifest_metadata import check_contract

_IMAGE = Path(__file__).parent.parent / "docs" / "images" / "01-diffuse-sdxl-turbo.png"


def _fresh_process(code: str) -> list[str]:
    """Runs SDK code in its own interpreter so earlier GPU models can't starve it."""
    done = subprocess.run(
        [sys.executable, "-c", f"import splat\n{code}"], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.split()


def _sharp_orbit(image):
    """mlx3d-capture needs overlapping views; SHARP orbit renders supply them."""
    frames = _fresh_process(
        f"cloud = splat.gaussian({str(image)!r})[0]\n"
        "for a in (-10, -4, 4, 10):\n"
        "    print(splat.render(cloud, azimuth=a, width=512, height=384, engine='eevee')[0].id)"
    )
    ids = _fresh_process(
        f"for m in splat.gaussian([splat.Manifest.load(i) for i in {frames!r}], "
        "model='mlx3d-capture', quality='fast'):\n    print(m.id)"
    )
    return [splat.Manifest.load(i) for i in ids]


SMOKE = {
    "sdxl-turbo-mlx": lambda image: [splat.diffuse("a red chair", steps=1).asset],
    "sd21-coreml": lambda image: [splat.diffuse("a red chair", model="sd21-coreml", steps=2).asset],
    "fastvlm-0.5b": lambda image: splat.caption(image),
    "mobileclip2-s0": lambda image: splat.embed(image) + splat.embed(text="a chair"),
    "sam-mlx": lambda image: splat.segment(image, max_stickers=2),
    "sam2-coreml": lambda image: splat.segment(image, model="sam2-coreml", max_stickers=2),
    "depth-pro": lambda image: splat.depth(image),
    "depth-anything-v2-coreml": lambda image: splat.depth(image, model="depth-anything-v2-coreml"),
    "realesrgan-mlx": lambda image: splat.upscale(image, factor=2),
    "sharp": lambda image: splat.gaussian(image),
    "mlx3d-capture": _sharp_orbit,
}


def test_every_catalogued_model_has_a_smoke_test():
    from splat.application.models_admin import _all_catalogs

    assert set(_all_catalogs()) == set(SMOKE)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("SPLAT_INTEGRATION_TESTS") != "1",
    reason="set SPLAT_INTEGRATION_TESTS=1 to run every model for real",
)
@pytest.mark.parametrize("model", sorted(SMOKE))
def test_model_output_meets_its_kind_contract(model, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_MANIFEST_CACHE_DIR", str(tmp_path / "cache"))

    outputs = SMOKE[model](_IMAGE)

    produced = [m for m in outputs if m.created_by.endswith(f":{model}")]
    assert produced, f"{model} produced nothing"
    for manifest in produced:
        assert manifest.kind is not ManifestKind.FAN_OUT
        check_contract(manifest.kind, manifest.metadata, manifest.content_path.read_bytes())
