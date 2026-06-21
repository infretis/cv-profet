from pathlib import Path

import cv_analyze
import cv_builder


def test_cv_builder_resolves_from_environment(monkeypatch, tmp_path):
    (tmp_path / "CV_builder").mkdir()
    (tmp_path / "CV_builder" / "main.py").write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setenv("CV_PROFET_ROOT", str(tmp_path))
    assert cv_builder._resolve_repo_root() == tmp_path.resolve()


def test_cv_analyze_resolves_from_environment(monkeypatch, tmp_path):
    (tmp_path / "ppa").mkdir()
    (tmp_path / "ppa" / "analyze.py").write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setenv("CV_PROFET_ROOT", str(tmp_path))
    assert cv_analyze._resolve_repo_root() == tmp_path.resolve()


def test_cv_builder_main_returns_2_if_repo_not_found(monkeypatch, capsys):
    monkeypatch.setattr(cv_builder, "_resolve_repo_root", lambda: None)
    rc = cv_builder.main()
    err = capsys.readouterr().err

    assert rc == 2
    assert "could not locate cv-profet root" in err


def test_cv_analyze_main_returns_2_if_repo_not_found(monkeypatch, capsys):
    monkeypatch.setattr(cv_analyze, "_resolve_repo_root", lambda: None)
    rc = cv_analyze.main()
    err = capsys.readouterr().err

    assert rc == 2
    assert "could not locate cv-profet root" in err


def test_cv_builder_main_invokes_subprocess(monkeypatch, tmp_path):
    entry = tmp_path / "CV_builder" / "main.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setattr(cv_builder, "_resolve_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cv_builder.sys, "argv", ["cv_builder", "--toml", "infretis.toml"])

    called = {}

    def fake_call(cmd, cwd):
        called["cmd"] = cmd
        called["cwd"] = cwd
        return 0

    monkeypatch.setattr(cv_builder.subprocess, "call", fake_call)
    rc = cv_builder.main()

    assert rc == 0
    assert Path(called["cmd"][1]) == entry
    assert called["cmd"][2:] == ["--toml", "infretis.toml"]
    assert called["cwd"] == str(Path.cwd())


def test_cv_analyze_main_invokes_subprocess(monkeypatch, tmp_path):
    entry = tmp_path / "ppa" / "analyze.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setattr(cv_analyze, "_resolve_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cv_analyze.sys, "argv", ["cv_analyze", "--screen", "--toml", "pp.toml"])

    called = {}

    def fake_call(cmd, cwd):
        called["cmd"] = cmd
        called["cwd"] = cwd
        return 0

    monkeypatch.setattr(cv_analyze.subprocess, "call", fake_call)
    rc = cv_analyze.main()

    assert rc == 0
    assert Path(called["cmd"][1]) == entry
    assert called["cmd"][2:] == ["--screen", "--toml", "pp.toml"]
    assert called["cwd"] == str(Path.cwd())
