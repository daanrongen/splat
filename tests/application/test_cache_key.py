from unittest.mock import patch

from splat.application import pipeline
from splat.application.pipeline import compute_cache_key


def _key(stage: str) -> str:
    return compute_cache_key(stage=stage, model="m", params={"a": 1})


def test_bumping_a_stage_revision_changes_only_that_stage():
    before = {s: _key(s) for s in ("mesh", "depth")}
    with patch.dict(pipeline.STAGE_REVISION, {"mesh": pipeline.STAGE_REVISION["mesh"] + 1}):
        assert _key("mesh") != before["mesh"]
        assert _key("depth") == before["depth"]


def test_unrevised_stage_key_is_unchanged():
    with patch.dict(pipeline.STAGE_REVISION, {}, clear=True):
        assert _key("depth") == compute_cache_key(stage="depth", model="m", params={"a": 1})
