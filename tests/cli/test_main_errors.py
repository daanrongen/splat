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
