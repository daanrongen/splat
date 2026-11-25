from splat.application.models_admin import PruneModelsUseCase

SAM = "facebook/sam-vit-base"


class FakeModelSource:
    """`facebook/sam-vit-base` is cataloged (sam-mlx); the other two are not."""

    def __init__(self, *, stored=(SAM,), legacy=None) -> None:
        self.removed: list[str] = []
        self.removed_legacy: list[str] = []
        self._stored = set(stored)
        self._legacy = legacy or {}

    def list_cached(self) -> list[str]:
        return ["dropped-org/dropped-model", SAM, "some-org/leftover"]

    def is_cached(self, model_id: str, files: tuple[str, ...] = ()) -> bool:
        return model_id in self._stored

    def size_on_disk(self, model_id: str) -> int:
        return {"dropped-org/dropped-model": 444_000_000, "some-org/leftover": 1_000}.get(
            model_id, 0
        )

    def list_legacy(self) -> dict[str, int]:
        return self._legacy

    def remove(self, model_id: str) -> None:
        self.removed.append(model_id)

    def remove_legacy(self, model_id: str) -> None:
        self.removed_legacy.append(model_id)


def test_find_reports_only_uncataloged_repos():
    orphans = PruneModelsUseCase(FakeModelSource()).find()

    assert [orphan.repo_id for orphan in orphans] == [
        "dropped-org/dropped-model",
        "some-org/leftover",
    ]
    assert orphans[0].size_bytes == 444_000_000


def test_find_does_not_delete():
    source = FakeModelSource(legacy={SAM: 5})
    PruneModelsUseCase(source).find()
    assert source.removed == [] and source.removed_legacy == []


def test_execute_removes_every_orphan_and_leaves_cataloged_weights():
    source = FakeModelSource()

    removed = PruneModelsUseCase(source).execute()

    assert source.removed == ["dropped-org/dropped-model", "some-org/leftover"]
    assert "facebook/sam-vit-base" not in source.removed
    assert [orphan.repo_id for orphan in removed] == source.removed


def test_find_reports_a_hub_copy_the_store_already_holds():
    orphans = PruneModelsUseCase(FakeModelSource(legacy={SAM: 1_000_000_000})).find()

    assert [(o.repo_id, o.legacy) for o in orphans if o.legacy] == [(SAM, True)]


def test_find_keeps_a_hub_copy_the_store_does_not_hold():
    orphans = PruneModelsUseCase(FakeModelSource(stored=(), legacy={SAM: 5})).find()

    assert not [o for o in orphans if o.legacy]


def test_find_leaves_uncataloged_hub_repos_alone():
    orphans = PruneModelsUseCase(FakeModelSource(legacy={"other/tool-model": 5})).find()

    assert not [o for o in orphans if o.legacy]


def test_execute_removes_hub_copies_through_the_legacy_path():
    source = FakeModelSource(legacy={SAM: 5})

    PruneModelsUseCase(source).execute()

    assert source.removed_legacy == [SAM]
    assert SAM not in source.removed
