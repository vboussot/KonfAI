import tempfile
from contextlib import nullcontext
from pathlib import Path

import pytest

import konfai.app as app_module


def test_run_distributed_app_uses_requested_workspace_and_restores_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    run_dir = tmp_path / "workspace"
    user_dir.mkdir()
    run_dir.mkdir()
    monkeypatch.chdir(user_dir)
    monkeypatch.setattr(app_module, "MinimalLog", nullcontext)

    visited: dict[str, Path] = {}

    @app_module.run_distributed_app
    def wrapped(tmp_dir: Path) -> None:
        visited["cwd"] = Path.cwd()
        (Path.cwd() / "result.txt").write_text("ok", encoding="utf-8")

    wrapped(tmp_dir=run_dir)

    assert visited["cwd"] == run_dir
    assert Path.cwd() == user_dir
    assert (run_dir / "result.txt").read_text(encoding="utf-8") == "ok"


def test_run_distributed_app_cleans_auto_created_temporary_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    auto_dir = Path(tempfile.gettempdir()) / "konfai_test_auto_workspace"
    auto_dir.mkdir(exist_ok=True)
    user_dir.mkdir()
    monkeypatch.chdir(user_dir)
    monkeypatch.setattr(app_module, "MinimalLog", nullcontext)
    monkeypatch.setattr(app_module.tempfile, "mkdtemp", lambda prefix: str(auto_dir))

    visited: dict[str, Path] = {}

    @app_module.run_distributed_app
    def wrapped() -> None:
        visited["cwd"] = Path.cwd()
        (Path.cwd() / "result.txt").write_text("ok", encoding="utf-8")

    wrapped()

    assert visited["cwd"] == auto_dir
    assert Path.cwd() == user_dir
    assert auto_dir.exists() is False


def test_run_distributed_app_restores_cwd_after_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    user_dir = tmp_path / "user"
    run_dir = tmp_path / "workspace"
    user_dir.mkdir()
    run_dir.mkdir()
    monkeypatch.chdir(user_dir)
    monkeypatch.setattr(app_module, "MinimalLog", nullcontext)

    @app_module.run_distributed_app
    def wrapped(tmp_dir: Path) -> None:
        raise KeyboardInterrupt

    wrapped(tmp_dir=run_dir)

    assert Path.cwd() == user_dir
    assert "Manual interruption" in capsys.readouterr().out
