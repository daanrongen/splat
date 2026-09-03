"""Wiring seam for mesh-export backends. Empty until a real one lands — see
`ports/mesh.py` and `cli/mesh.py` for why this is still a stub.
"""

from dataclasses import dataclass

from splat.ports.mesh import MeshExporter


@dataclass(frozen=True)
class MeshModelDescriptor:
    name: str
    backend_cls: type[MeshExporter]


MESH_CATALOG: dict[str, MeshModelDescriptor] = {}
