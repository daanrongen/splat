from splat.application.models_admin import PruneModelsUseCase


class FakeModelSource:
    """`facebook/sam-vit-base` is cataloged (sam-mlx); the other two are not."""

    def __init__(self) -> None:
        self.removed: list[str] = []

    def list_cached(self) -> list[str]:
        return ["dropped-org/dropped-model", "facebook/sam-vit-base", "some-org/leftover"]

    def size_on_disk(self, model_id: str) -> int:
        return {"dropped-org/dropped-model": 444_000_000, "some-org/leftover": 1_000}.get(
            model_id, 0
        )

    def remove(self, model_id: str) -> None:
        self.removed.append(model_id)


def test_find_reports_only_uncataloged_repos():
    orphans = PruneModelsUseCase(FakeModelSource()).find()

    assert [orphan.repo_id for orphan in orphans] == [
        "dropped-org/dropped-model",
        "some-org/leftover",
    ]
    assert orphans[0].size_bytes == 444_000_000


def test_find_does_not_delete():
    source = FakeModelSource()
    PruneModelsUseCase(source).find()
    assert source.removed == []


def test_execute_removes_every_orphan_and_leaves_cataloged_weights():
    source = FakeModelSource()

    removed = PruneModelsUseCase(source).execute()

    assert source.removed == ["dropped-org/dropped-model", "some-org/leftover"]
    assert "facebook/sam-vit-base" not in source.removed
    assert [orphan.repo_id for orphan in removed] == source.removed
