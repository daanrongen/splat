from splat.adapters.model_sources.huggingface import HuggingFaceModelSource


class FakeRevision:
    def __init__(self, snapshot_path) -> None:
        self.snapshot_path = snapshot_path


class FakeRepo:
    def __init__(self, repo_id: str, revisions=()) -> None:
        self.repo_id = repo_id
        self.revisions = list(revisions)


class FakeCacheInfo:
    def __init__(self, repos) -> None:
        self.repos = list(repos)


def test_pull_calls_snapshot_download(mocker, tmp_path):
    mock_download = mocker.patch(
        "splat.adapters.model_sources.huggingface.snapshot_download",
        return_value=str(tmp_path),
    )

    result = HuggingFaceModelSource().pull("some/model")

    assert result == tmp_path
    mock_download.assert_called_once()
    assert mock_download.call_args.kwargs["repo_id"] == "some/model"
    assert "cache_dir" not in mock_download.call_args.kwargs


def test_is_cached_true_when_scan_cache_dir_finds_repo(mocker, tmp_path):
    mocker.patch(
        "splat.adapters.model_sources.huggingface.scan_cache_dir",
        return_value=FakeCacheInfo([FakeRepo("some/model", [FakeRevision(tmp_path)])]),
    )
    assert HuggingFaceModelSource().is_cached("some/model") is True


def test_is_cached_false_when_not_found_anywhere(mocker):
    mocker.patch(
        "splat.adapters.model_sources.huggingface.scan_cache_dir",
        return_value=FakeCacheInfo([]),
    )
    assert HuggingFaceModelSource().is_cached("some/model") is False


def test_is_cached_true_for_bespoke_local_dir_layout(mocker, tmp_path):
    mocker.patch(
        "splat.adapters.model_sources.huggingface.scan_cache_dir",
        return_value=FakeCacheInfo([]),
    )
    mocker.patch(
        "splat.adapters.model_sources.huggingface.model_cache_dir",
        return_value=tmp_path,
    )
    bespoke_dir = tmp_path / "coreml-stable-diffusion" / "apple--coreml-stable-diffusion-2-1-base"
    bespoke_dir.mkdir(parents=True)
    (bespoke_dir / "some_file.json").write_text("{}")

    assert HuggingFaceModelSource().is_cached("apple/coreml-stable-diffusion-2-1-base") is True


def test_is_cached_false_for_empty_bespoke_local_dir(mocker, tmp_path):
    mocker.patch(
        "splat.adapters.model_sources.huggingface.scan_cache_dir",
        return_value=FakeCacheInfo([]),
    )
    mocker.patch(
        "splat.adapters.model_sources.huggingface.model_cache_dir",
        return_value=tmp_path,
    )
    (tmp_path / "coreml-stable-diffusion" / "apple--coreml-stable-diffusion-2-1-base").mkdir(
        parents=True
    )

    assert HuggingFaceModelSource().is_cached("apple/coreml-stable-diffusion-2-1-base") is False


def test_remove_deletes_model_root(mocker, tmp_path):
    snapshot_dir = tmp_path / "models--org--name" / "snapshots" / "rev"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "weights.bin").write_bytes(b"data")

    mocker.patch(
        "splat.adapters.model_sources.huggingface.scan_cache_dir",
        return_value=FakeCacheInfo([FakeRepo("org/name", [FakeRevision(snapshot_dir)])]),
    )

    HuggingFaceModelSource().remove("org/name")

    assert not (tmp_path / "models--org--name").exists()


def test_remove_deletes_bespoke_local_dir_without_deleting_model_cache(mocker, tmp_path):
    mocker.patch(
        "splat.adapters.model_sources.huggingface.scan_cache_dir",
        return_value=FakeCacheInfo([]),
    )
    mocker.patch(
        "splat.adapters.model_sources.huggingface.model_cache_dir",
        return_value=tmp_path,
    )
    bespoke_dir = tmp_path / "coreml-stable-diffusion" / "apple--coreml-stable-diffusion-2-1-base"
    bespoke_dir.mkdir(parents=True)
    (bespoke_dir / "some_file.json").write_text("{}")

    HuggingFaceModelSource().remove("apple/coreml-stable-diffusion-2-1-base")

    assert tmp_path.exists()
    assert not bespoke_dir.exists()


def test_remove_noop_when_not_cached(mocker):
    mocker.patch(
        "splat.adapters.model_sources.huggingface.scan_cache_dir",
        return_value=FakeCacheInfo([]),
    )
    HuggingFaceModelSource().remove("org/name")  # should not raise


def test_list_cached(mocker, tmp_path):
    mocker.patch("splat.adapters.model_sources.huggingface.model_cache_dir", return_value=tmp_path)
    mocker.patch(
        "splat.adapters.model_sources.huggingface.scan_cache_dir",
        return_value=FakeCacheInfo([FakeRepo("org/a"), FakeRepo("org/b")]),
    )
    assert HuggingFaceModelSource().list_cached() == ["org/a", "org/b"]


def test_list_cached_includes_bespoke_converted_dirs(mocker, tmp_path):
    (tmp_path / "coreml-sam2" / "apple--coreml-sam2.1-tiny").mkdir(parents=True)
    (tmp_path / "coreml-sam2" / "apple--coreml-sam2.1-tiny" / "w.bin").write_bytes(b"x")
    mocker.patch("splat.adapters.model_sources.huggingface.model_cache_dir", return_value=tmp_path)
    mocker.patch(
        "splat.adapters.model_sources.huggingface.scan_cache_dir",
        return_value=FakeCacheInfo([FakeRepo("org/a")]),
    )

    assert HuggingFaceModelSource().list_cached() == [
        "apple/coreml-sam2.1-tiny",
        "org/a",
    ]


def test_size_on_disk_counts_blobs_once_not_through_snapshot_symlinks(mocker, tmp_path):
    repo = tmp_path / "models--org--name"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (blobs / "deadbeef").write_bytes(b"w" * 1024)
    (snapshot / "model.safetensors").symlink_to(blobs / "deadbeef")

    mocker.patch(
        "splat.adapters.model_sources.huggingface.scan_cache_dir",
        return_value=FakeCacheInfo([FakeRepo("org/name", [FakeRevision(snapshot)])]),
    )

    assert HuggingFaceModelSource().size_on_disk("org/name") == 1024


def test_size_on_disk_zero_when_not_cached(mocker):
    mocker.patch(
        "splat.adapters.model_sources.huggingface.scan_cache_dir",
        return_value=FakeCacheInfo([]),
    )
    assert HuggingFaceModelSource().size_on_disk("org/name") == 0
