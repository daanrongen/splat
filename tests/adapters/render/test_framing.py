import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[3] / "src/splat/adapters/render/_blender_script.py"
TAN_HALF = np.tan(np.radians(40) / 2)


@pytest.fixture
def script(monkeypatch):
    for name in ("bpy", "mathutils"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["mathutils"].Matrix = sys.modules["mathutils"].Vector = lambda *_: None
    spec = importlib.util.spec_from_file_location("blender_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sphere_with_rod(scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    sphere = rng.normal(size=(2000, 3))
    sphere = sphere / np.linalg.norm(sphere, axis=1, keepdims=True) * rng.uniform(0, 1, (2000, 1))
    rod = np.stack([np.linspace(1, 3, 300), np.zeros(300), np.zeros(300)], axis=1)
    return np.concatenate([sphere, rod]) * scale


def _off_axis_tangent(means: np.ndarray, center: np.ndarray, distance: float) -> float:
    camera = center + distance * np.array([0.0, 0.0, 1.0])
    rays = means - camera
    return float((np.linalg.norm(rays[:, :2], axis=1) / -rays[:, 2]).max())


def test_default_framing_keeps_a_thin_protrusion_in_frame(script):
    means = _sphere_with_rod()
    center, radius = script._bounding_sphere(means)
    distance = script._fit_distance(radius, TAN_HALF, 1.0)
    assert _off_axis_tangent(means, center, distance) <= TAN_HALF


def test_zoom_scales_the_distance_independent_of_units(script):
    ratios = []
    for scale in (1.0, 0.3, 40.0):
        _, radius = script._bounding_sphere(_sphere_with_rod(scale))
        fit = script._fit_distance(radius, TAN_HALF, 1.0)
        assert script._fit_distance(radius, TAN_HALF, 2.0) == pytest.approx(fit / 2)
        ratios.append(fit / scale)
    assert ratios == pytest.approx([ratios[0]] * 3)


def test_fov_is_measured_along_the_narrower_side(script):
    cam = SimpleNamespace(angle=np.radians(40), sensor_fit="AUTO")
    assert script._tan_half_fov(cam, 1000, 1000) == pytest.approx(TAN_HALF)
    assert script._tan_half_fov(cam, 2000, 1000) == pytest.approx(TAN_HALF / 2)
    assert script._tan_half_fov(cam, 1000, 2000) == pytest.approx(TAN_HALF / 2)
