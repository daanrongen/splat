from splat.adapters.model_sources.huggingface import HuggingFaceModelSource


def test_pull_calls_snapshot_download(mocker, tmp_path):
    mock_download = mocker.patch(
        "splat.adapters.model_sources.huggingface.snapshot_download",
        return_value=str(tmp_path),
    )

    result = HuggingFaceModelSource().pull("some/model")

    assert result == tmp_path
    mock_download.assert_called_once()
    assert mock_download.call_args.kwargs["repo_id"] == "some/model"


def test_is_cached_true_when_local_files_only_succeeds(mocker, tmp_path):
    mocker.patch(
        "splat.adapters.model_sources.huggingface.snapshot_download",
        return_value=str(tmp_path),
    )
    assert HuggingFaceModelSource().is_cached("some/model") is True


def test_is_cached_false_when_snapshot_download_raises(mocker):
    mocker.patch(
        "splat.adapters.model_sources.huggingface.snapshot_download",
        side_effect=Exception("not cached"),
    )
    assert HuggingFaceModelSource().is_cached("some/model") is False


def test_remove_deletes_model_root(mocker, tmp_path):
    snapshot_dir = tmp_path / "models--org--name" / "snapshots" / "rev"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "weights.bin").write_bytes(b"data")

    mocker.patch(
        "splat.adapters.model_sources.huggingface.snapshot_download",
        return_value=str(snapshot_dir),
    )

    HuggingFaceModelSource().remove("org/name")

    assert not (tmp_path / "models--org--name").exists()


def test_remove_noop_when_not_cached(mocker):
    mocker.patch(
        "splat.adapters.model_sources.huggingface.snapshot_download",
        side_effect=Exception("not cached"),
    )
    HuggingFaceModelSource().remove("org/name")  # should not raise


def test_list_cached(mocker):
    class FakeRepo:
        def __init__(self, repo_id: str) -> None:
            self.repo_id = repo_id

    class FakeCacheInfo:
        def __init__(self) -> None:
            self.repos = [FakeRepo("org/a"), FakeRepo("org/b")]

    mocker.patch(
        "splat.adapters.model_sources.huggingface.scan_cache_dir",
        return_value=FakeCacheInfo(),
    )
    assert HuggingFaceModelSource().list_cached() == ["org/a", "org/b"]
