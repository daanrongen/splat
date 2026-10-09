import pytest

from splat.adapters.model_sources.huggingface import HuggingFaceModelSource


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("SPLAT_MODEL_CACHE_DIR", str(tmp_path / "models"))
    return tmp_path / "hub"


def add_snapshot(hub, repo_id: str):
    snapshot = hub / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    return snapshot


def add_bespoke(tmp_path, kind: str, repo_id: str, *, empty: bool = False):
    directory = tmp_path / "models" / kind / repo_id.replace("/", "--")
    directory.mkdir(parents=True)
    if not empty:
        (directory / "some_file.json").write_text("{}")
    return directory


def test_pull_calls_snapshot_download(mocker, tmp_path):
    mock_download = mocker.patch("huggingface_hub.snapshot_download", return_value=str(tmp_path))

    result = HuggingFaceModelSource().pull("some/model")

    assert result == tmp_path
    mock_download.assert_called_once()
    assert mock_download.call_args.kwargs["repo_id"] == "some/model"
    assert "cache_dir" not in mock_download.call_args.kwargs


def test_is_cached_true_when_hub_snapshot_exists(hub):
    add_snapshot(hub, "some/model")
    assert HuggingFaceModelSource().is_cached("some/model") is True


def test_is_cached_false_when_not_found_anywhere(hub):
    assert HuggingFaceModelSource().is_cached("some/model") is False


def test_is_cached_false_for_repo_dir_without_snapshots(hub):
    (hub / "models--some--model").mkdir(parents=True)
    assert HuggingFaceModelSource().is_cached("some/model") is False


def test_is_cached_true_for_bespoke_local_dir_layout(hub, tmp_path):
    add_bespoke(tmp_path, "coreml-stable-diffusion", "apple/coreml-stable-diffusion-2-1-base")
    assert HuggingFaceModelSource().is_cached("apple/coreml-stable-diffusion-2-1-base") is True


def test_is_cached_false_for_empty_bespoke_local_dir(hub, tmp_path):
    add_bespoke(
        tmp_path, "coreml-stable-diffusion", "apple/coreml-stable-diffusion-2-1-base", empty=True
    )
    assert HuggingFaceModelSource().is_cached("apple/coreml-stable-diffusion-2-1-base") is False


def test_remove_deletes_model_root(hub):
    snapshot = add_snapshot(hub, "org/name")
    (snapshot / "weights.bin").write_bytes(b"data")

    HuggingFaceModelSource().remove("org/name")

    assert not (hub / "models--org--name").exists()


def test_remove_deletes_bespoke_local_dir_without_deleting_model_cache(hub, tmp_path):
    bespoke = add_bespoke(
        tmp_path, "coreml-stable-diffusion", "apple/coreml-stable-diffusion-2-1-base"
    )

    HuggingFaceModelSource().remove("apple/coreml-stable-diffusion-2-1-base")

    assert (tmp_path / "models").exists()
    assert not bespoke.exists()


def test_remove_noop_when_not_cached(hub):
    HuggingFaceModelSource().remove("org/name")  # should not raise


def test_list_cached(hub):
    add_snapshot(hub, "org/a")
    add_snapshot(hub, "org/b")
    assert HuggingFaceModelSource().list_cached() == ["org/a", "org/b"]


def test_list_cached_includes_bespoke_converted_dirs(hub, tmp_path):
    add_bespoke(tmp_path, "coreml-sam2", "apple/coreml-sam2.1-tiny")
    add_snapshot(hub, "org/a")

    assert HuggingFaceModelSource().list_cached() == ["apple/coreml-sam2.1-tiny", "org/a"]


def test_size_on_disk_counts_blobs_once_not_through_snapshot_symlinks(hub):
    repo = hub / "models--org--name"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "deadbeef").write_bytes(b"w" * 1024)
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to(blobs / "deadbeef")

    assert HuggingFaceModelSource().size_on_disk("org/name") == 1024


def test_size_on_disk_zero_when_not_cached(hub):
    assert HuggingFaceModelSource().size_on_disk("org/name") == 0
