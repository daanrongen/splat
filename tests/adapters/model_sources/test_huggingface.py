import pytest

from splat.adapters.model_sources.huggingface import HuggingFaceModelSource


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("SPLAT_MODEL_CACHE_DIR", str(tmp_path / "models"))
    return tmp_path / "hub"


def add_hub_file(hub, repo_id: str, name: str, data: bytes = b"w"):
    repo = hub / f"models--{repo_id.replace('/', '--')}"
    blob = repo / "blobs" / name.replace("/", "_")
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)
    link = repo / "snapshots" / "rev" / name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(blob)


def add_stored(tmp_path, repo_id: str, name: str = "weights.bin", data: bytes = b"w"):
    path = tmp_path / "models" / repo_id.replace("/", "--") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_pull_downloads_into_the_store_as_real_files(hub, mocker, tmp_path):
    mock_download = mocker.patch("huggingface_hub.snapshot_download")

    result = HuggingFaceModelSource().pull("some/model")

    assert result == tmp_path / "models" / "some--model"
    assert mock_download.call_args.kwargs["local_dir"] == result
    assert "cache_dir" not in mock_download.call_args.kwargs


def test_pull_asks_only_for_the_files_a_model_needs(hub, mocker):
    mock_download = mocker.patch("huggingface_hub.snapshot_download")

    HuggingFaceModelSource().pull("some/model", files=("*.json", "unet/*.safetensors"))

    assert mock_download.call_args.kwargs["allow_patterns"] == ["*.json", "unet/*.safetensors"]


def test_pull_of_a_whole_repo_sets_no_pattern(hub, mocker):
    mock_download = mocker.patch("huggingface_hub.snapshot_download")

    HuggingFaceModelSource().pull("some/model")

    assert mock_download.call_args.kwargs["allow_patterns"] is None


def test_pull_skips_the_network_when_the_files_are_present(hub, mocker, tmp_path):
    add_stored(tmp_path, "some/model", "config.json")
    mock_download = mocker.patch("huggingface_hub.snapshot_download")

    HuggingFaceModelSource().pull("some/model", files=("*.json",))

    mock_download.assert_not_called()


def test_pull_hardlinks_files_from_the_hub_cache(hub, mocker, tmp_path):
    add_hub_file(hub, "some/model", "config.json", b"{}")
    mock_download = mocker.patch("huggingface_hub.snapshot_download")

    HuggingFaceModelSource().pull("some/model", files=("config.json",))

    stored = tmp_path / "models" / "some--model" / "config.json"
    assert stored.read_bytes() == b"{}"
    assert stored.stat().st_nlink == 2
    mock_download.assert_not_called()


def test_pull_adopts_only_the_files_asked_for(hub, mocker, tmp_path):
    add_hub_file(hub, "some/model", "config.json")
    add_hub_file(hub, "some/model", "pytorch_model.bin")
    mocker.patch("huggingface_hub.snapshot_download")

    HuggingFaceModelSource().pull("some/model", files=("config.json",))

    assert not (tmp_path / "models" / "some--model" / "pytorch_model.bin").exists()


def test_pull_downloads_what_the_hub_cache_lacks(hub, mocker):
    add_hub_file(hub, "some/model", "config.json")
    mock_download = mocker.patch("huggingface_hub.snapshot_download")

    HuggingFaceModelSource().pull("some/model", files=("config.json", "model.safetensors"))

    mock_download.assert_called_once()


def test_is_cached_needs_every_pattern_to_match(hub, tmp_path):
    add_stored(tmp_path, "some/model", "config.json")

    assert HuggingFaceModelSource().is_cached("some/model", ("*.json", "*.safetensors")) is False


def test_is_cached_matches_files_below_a_package_directory(hub, tmp_path):
    add_stored(tmp_path, "some/model", "a.mlpackage/Data/weights/weight.bin")

    assert HuggingFaceModelSource().is_cached("some/model", ("a.mlpackage/**",)) is True


def test_is_cached_ignores_the_download_metadata_directory(hub, tmp_path):
    add_stored(tmp_path, "some/model", ".cache/huggingface/download/x.metadata")

    assert HuggingFaceModelSource().is_cached("some/model") is False


def test_is_cached_false_when_not_in_the_store(hub):
    assert HuggingFaceModelSource().is_cached("some/model") is False


def test_local_path_is_the_store_directory(hub, tmp_path):
    add_stored(tmp_path, "some/model")

    assert HuggingFaceModelSource().local_path("some/model") == tmp_path / "models" / "some--model"


def test_remove_deletes_the_model_directory(hub, tmp_path):
    stored = add_stored(tmp_path, "some/model")

    HuggingFaceModelSource().remove("some/model")

    assert not stored.parent.exists()


def test_remove_noop_when_not_cached(hub):
    HuggingFaceModelSource().remove("some/model")  # should not raise


def test_list_cached_names_every_store_directory(hub, tmp_path):
    add_stored(tmp_path, "org/a")
    add_stored(tmp_path, "mlx-sam", "old/model.safetensors")

    assert HuggingFaceModelSource().list_cached() == ["mlx-sam", "org/a"]


def test_size_on_disk_sums_the_files(hub, tmp_path):
    add_stored(tmp_path, "some/model", "a.bin", b"w" * 1024)
    add_stored(tmp_path, "some/model", "b/c.bin", b"w" * 24)

    assert HuggingFaceModelSource().size_on_disk("some/model") == 1048


def test_size_on_disk_zero_when_not_cached(hub):
    assert HuggingFaceModelSource().size_on_disk("some/model") == 0


def test_list_legacy_counts_hub_blobs_once(hub):
    add_hub_file(hub, "org/name", "model.safetensors", b"w" * 1024)

    assert HuggingFaceModelSource().list_legacy() == {"org/name": 1024}


def test_list_legacy_skips_repos_without_a_snapshot(hub):
    (hub / "models--org--name").mkdir(parents=True)

    assert HuggingFaceModelSource().list_legacy() == {}


def test_remove_legacy_deletes_the_hub_repo(hub):
    add_hub_file(hub, "org/name", "model.safetensors")

    HuggingFaceModelSource().remove_legacy("org/name")

    assert not (hub / "models--org--name").exists()


def _bar_class(mocker, hub, reports):
    mock_download = mocker.patch("huggingface_hub.snapshot_download")
    HuggingFaceModelSource().pull("some/model", on_progress=lambda *report: reports.append(report))
    return mock_download.call_args.kwargs["tqdm_class"]


def test_pull_without_a_callback_passes_no_progress_class(mocker, hub):
    mock_download = mocker.patch("huggingface_hub.snapshot_download")

    HuggingFaceModelSource().pull("some/model")

    assert "tqdm_class" not in mock_download.call_args.kwargs


def test_progress_reports_the_second_byte_bar(mocker, hub):
    reports = []
    bar_class = _bar_class(mocker, hub, reports)
    bar_class(total=0, unit="B")  # transfer bar
    reconstruct = bar_class(total=0, unit="B")

    reconstruct.total = 100
    reconstruct.update(40)
    reconstruct.update(60)

    assert reports == [(40, 100), (100, 100)]


def test_progress_ignores_the_transfer_bar(mocker, hub):
    reports = []
    bar_class = _bar_class(mocker, hub, reports)
    transfer = bar_class(total=0, unit="B")
    bar_class(total=0, unit="B")

    transfer.update(10)

    assert reports == []


def test_progress_ignores_the_file_count_bar(mocker, hub):
    reports = []
    bar_class = _bar_class(mocker, hub, reports)
    bar_class(total=0, unit="B")
    bar_class(total=0, unit="B")

    bar_class(total=3, unit="it").update(1)

    assert reports == []


def test_progress_survives_disabled_hub_bars(mocker, hub, monkeypatch):
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    reports = []
    bar_class = _bar_class(mocker, hub, reports)
    bar_class(total=0, unit="B")
    reconstruct = bar_class(total=50, unit="B")

    reconstruct.update(50)

    assert reports == [(50, 50)]
