import splat.env as env_module
from splat.env import resolve, resolve_verbose


def test_resolve_prefers_os_environ(monkeypatch):
    monkeypatch.setenv("SPLAT_DIFFUSE_MODEL", "from-env")
    assert resolve("SPLAT_DIFFUSE_MODEL", "default-value") == "from-env"


def test_resolve_verbose_reports_env_source(monkeypatch):
    monkeypatch.setenv("SPLAT_DIFFUSE_MODEL", "from-env")
    resolved = resolve_verbose("SPLAT_DIFFUSE_MODEL", "default-value")
    assert resolved == env_module.Resolved("from-env", "env")


def test_resolve_falls_back_to_mise(monkeypatch):
    monkeypatch.delenv("SPLAT_DIFFUSE_MODEL", raising=False)
    monkeypatch.setattr(env_module, "_mise_env_cache", {"SPLAT_DIFFUSE_MODEL": "from-mise"})
    resolved = resolve_verbose("SPLAT_DIFFUSE_MODEL", "default-value")
    assert resolved == env_module.Resolved("from-mise", "mise")


def test_resolve_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("SPLAT_DIFFUSE_MODEL", raising=False)
    monkeypatch.setattr(env_module, "_mise_env_cache", {})
    resolved = resolve_verbose("SPLAT_DIFFUSE_MODEL", "default-value")
    assert resolved == env_module.Resolved("default-value", "default")


def test_mise_env_returns_empty_when_mise_missing(monkeypatch):
    monkeypatch.setattr(env_module, "_mise_env_cache", None)
    monkeypatch.setattr(env_module.shutil, "which", lambda _name: None)
    assert env_module._mise_env() == {}


def test_mise_env_memoizes(monkeypatch):
    monkeypatch.setattr(env_module, "_mise_env_cache", {"X": "1"})
    calls = []
    monkeypatch.setattr(env_module.shutil, "which", lambda _name: calls.append(1) or "/bin/mise")
    assert env_module._mise_env() == {"X": "1"}
    assert calls == []
