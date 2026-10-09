import subprocess
import sys

import pytest

from splat.cli import main as cli_main


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def test_failure_prints_one_line_and_exits_non_zero(monkeypatch, capsys):
    monkeypatch.delenv("SPLAT_DEBUG", raising=False)
    monkeypatch.setattr(cli_main, "app", _boom)

    with pytest.raises(SystemExit) as exit_info:
        cli_main.main()

    err = capsys.readouterr().err
    assert exit_info.value.code == 1
    assert "RuntimeError: boom" in err
    assert "Traceback" not in err


def test_debug_env_lets_the_traceback_through(monkeypatch):
    monkeypatch.setenv("SPLAT_DEBUG", "1")
    monkeypatch.setattr(cli_main, "app", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        cli_main.main()


def _bars_disabled(monkeypatch, *args: str) -> str:
    for var in ("SPLAT_DEBUG", "HF_HUB_DISABLE_PROGRESS_BARS"):
        monkeypatch.delenv(var, raising=False)
    code = "import os, splat.cli.main; print(os.environ.get('HF_HUB_DISABLE_PROGRESS_BARS'))"
    return subprocess.run(
        [sys.executable, "-c", code, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_third_party_output_is_quiet_by_default(monkeypatch):
    assert _bars_disabled(monkeypatch) == "1"


def test_debug_keeps_third_party_output(monkeypatch):
    assert _bars_disabled(monkeypatch, "--debug") == "None"
