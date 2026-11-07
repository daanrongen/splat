"""A real pull of one tiny repo. Opt in with SPLAT_INTEGRATION_TESTS=1; needs network."""

import os

import pytest

from splat.adapters.model_sources.huggingface import HuggingFaceModelSource

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("SPLAT_INTEGRATION_TESTS") != "1",
        reason="set SPLAT_INTEGRATION_TESTS=1 to pull a real repo from the Hub",
    ),
]


def test_pull_reports_progress_up_to_the_repo_size(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    reports = []

    HuggingFaceModelSource().pull(
        "prajjwal1/bert-tiny", on_progress=lambda *report: reports.append(report)
    )

    assert reports[-1][0] == reports[-1][1] > 0
