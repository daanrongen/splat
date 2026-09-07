"""Contract for `splat render` - see registry/wiring.py's `get_render_backend`
for backend selection. Unlike the model-backed stages (gaussian, diffuse, ...)
there is exactly one render backend (blender) today and no HF weights to
catalog, so this skips the ModelDescriptor/catalog pattern those use - add
one if/when a second backend (e.g. a native mlx3d/gsplat rasterizer) shows up.
"""

from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind

RENDER_CONTRACT = StageContract(
    stage="render",
    inputs=(Requirement(name="gaussian_cloud", any_of_tags=frozenset({"splat_3d"})),),
    produces=ManifestKind.IMAGE,
)
