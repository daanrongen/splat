"""Contract for `splat blender` - see registry/wiring.py's `get_render_backend`
for backend selection. Unlike the model-backed stages (gaussian, mesh, ...)
there is exactly one render backend and no HF weights to catalog, so this
skips the ModelDescriptor/catalog pattern those use.
"""

from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind

BLENDER_RENDER_CONTRACT = StageContract(
    stage="blender",
    inputs=(Requirement(name="gaussian_cloud", any_of_tags=frozenset({"splat_3d"})),),
    produces=ManifestKind.IMAGE,
)
