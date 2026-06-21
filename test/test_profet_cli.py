import profet
import pytest


def test_profet_build_dispatch(monkeypatch):
    called = {}

    def fake_run(module_name, args):
        called["module"] = module_name
        called["args"] = args
        return 0

    monkeypatch.setattr(profet, "_run_module", fake_run)
    monkeypatch.setattr(profet.sys, "argv", ["profet", "build", "--toml", "infretis.toml"])

    rc = profet.main()
    assert rc == 0
    assert called["module"] == "cv_builder"
    assert called["args"] == ["--toml", "infretis.toml"]


def test_profet_screen_dispatch(monkeypatch):
    called = {}

    def fake_run(module_name, args):
        called["module"] = module_name
        called["args"] = args
        return 0

    monkeypatch.setattr(profet, "_run_module", fake_run)
    monkeypatch.setattr(profet.sys, "argv", ["profet", "screen", "--toml", "infretis.toml"])

    rc = profet.main()
    assert rc == 0
    assert called["module"] == "cv_analyze"
    assert called["args"] == ["--screen", "--toml", "infretis.toml"]


def test_profet_check_active_dispatch(monkeypatch):
    called = {}

    def fake_run(module_name, args):
        called["module"] = module_name
        called["args"] = args
        return 0

    monkeypatch.setattr(profet, "_run_module", fake_run)
    monkeypatch.setattr(profet.sys, "argv", ["profet", "check_active", "--toml", "infretis.toml"])

    rc = profet.main()
    assert rc == 0
    assert called["module"] == "cv_builder"
    assert called["args"] == ["--check-active", "--toml", "infretis.toml"]


def test_profet_diagnose_dispatch(monkeypatch):
    called = {}

    def fake_run(module_name, args):
        called["module"] = module_name
        called["args"] = args
        return 0

    monkeypatch.setattr(profet, "_run_module", fake_run)
    monkeypatch.setattr(profet.sys, "argv", ["profet", "diagnose", "--toml", "infretis.toml"])

    rc = profet.main()
    assert rc == 0
    assert called["module"] == "cv_analyze"
    assert called["args"] == ["--diagnose", "--toml", "infretis.toml"]


def test_profet_analyze_is_rejected(monkeypatch):
    monkeypatch.setattr(profet.sys, "argv", ["profet", "analyze", "--screen"])
    with pytest.raises(SystemExit) as exc:
        profet.main()
    assert exc.value.code == 2
