from splat.application.models_admin import ListModelsUseCase

DEPTH_PRO_2 = ("apple/DepthPro2-H", "apple/DepthPro2-focal")


class FakeModelSource:
    def __init__(self, sizes=None, stored=()) -> None:
        self.sizes = sizes or {}
        self.stored = list(stored)

    def is_cached(self, model_id: str, files: tuple[str, ...] = ()) -> bool:
        return model_id in self.sizes

    def size_on_disk(self, model_id: str) -> int:
        return self.sizes.get(model_id, 0)

    def list_cached(self) -> list[str]:
        return self.stored


def _row(rows, name):
    return next(row for row in rows if row.name == name)


def test_every_catalog_model_gets_a_row():
    rows = ListModelsUseCase(FakeModelSource()).execute()

    assert {"sd15-coreml", "depth-pro-2", "mlx3d-capture"} <= {row.name for row in rows}


def test_a_model_is_cached_when_every_repo_is():
    source = FakeModelSource(sizes=dict.fromkeys(DEPTH_PRO_2, 1))

    assert _row(ListModelsUseCase(source).execute(), "depth-pro-2").cached is True


def test_a_model_missing_one_repo_is_not_cached():
    source = FakeModelSource(sizes={DEPTH_PRO_2[0]: 1})

    assert _row(ListModelsUseCase(source).execute(), "depth-pro-2").cached is False


def test_size_adds_up_every_repo_of_the_model():
    source = FakeModelSource(sizes={DEPTH_PRO_2[0]: 3_000, DEPTH_PRO_2[1]: 90})

    assert _row(ListModelsUseCase(source).execute(), "depth-pro-2").size_bytes == 3_090


def test_a_model_without_weights_has_no_size():
    assert _row(ListModelsUseCase(FakeModelSource()).execute(), "mlx3d-capture").size_bytes == 0


def test_rows_carry_the_stage_and_license():
    row = _row(ListModelsUseCase(FakeModelSource()).execute(), "sd15-coreml")

    assert (row.stage, row.license, row.commercial) == ("diffuse", "OpenRAIL-M", True)


def test_store_directories_outside_the_catalog_follow_as_unmanaged_rows():
    source = FakeModelSource(sizes={"mlx-sam": 358}, stored=["mlx-sam", "apple/Sharp"])

    unmanaged = [row for row in ListModelsUseCase(source).execute() if not row.managed]

    assert [(row.name, row.size_bytes) for row in unmanaged] == [("mlx-sam", 358)]


def test_catalog_repos_in_the_store_are_not_unmanaged():
    source = FakeModelSource(stored=["apple/Sharp"])

    assert all(row.managed for row in ListModelsUseCase(source).execute())
