"""Contract for `splat tools normalize.color` - see
application/tools/normalize_color.py for the correction algorithm. A single
fixed algorithm with no model to select, like `registry/render.py`.
"""

from splat.domain.contracts import Requirement, StageContract
from splat.domain.manifest import ManifestKind

NORMALIZE_COLOR_CONTRACT = StageContract(
    stage="tools.normalize.color",
    inputs=(Requirement(name="images", any_of_tags=frozenset({"colorlike"}), max_count=None),),
    produces=ManifestKind.IMAGE,
)
