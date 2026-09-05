from pathlib import Path

import numpy as np

from splat.domain.image_space import Shape3D
from splat.domain.manifest import ManifestKind
from splat.domain.value_objects import MIT
from splat.handlers.mesh import MeshRequest, handle
from splat.registry.wiring import get_asset_cache
from tests.image_helpers import write_sample_png


class FakeMeshBackend:
    name = "triposr"
    license = MIT

    def predict(self, image_path, **params) -> Shape3D:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        return Shape3D(vertices=vertices, faces=faces, metadata={"face_count": 2})


def _sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    return write_sample_png(path, (4, 4))


def test_handle_predicts_mesh_for_each_input(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_ASSET_CACHE_DIR", str(tmp_path / "cache"))
    mocker.patch("splat.handlers.mesh.get_mesh_backend", return_value=FakeMeshBackend())
    cache = get_asset_cache()
    asset = cache.put_external(_sample_image(tmp_path), kind=ManifestKind.IMAGE)

    results = handle(MeshRequest(inputs=[asset]))

    assert len(results) == 1
    assert results[0].kind == ManifestKind.SHAPE_3D
    assert results[0].created_by == "mesh:triposr"
