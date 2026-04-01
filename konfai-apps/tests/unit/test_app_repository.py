import json
from pathlib import Path

import pytest
from konfai_apps import app_repository as app_repository_module

from konfai.utils.errors import AppMetadataError


def test_get_app_repository_info_rejects_missing_required_metadata_keys(tmp_path: Path) -> None:
    app_dir = tmp_path / "broken_app"
    app_dir.mkdir()
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Broken App",
                "short_description": "Missing full description",
                "tta": 0,
                "mc_dropout": 0,
            }
        ),
        encoding="utf-8",
    )

    try:
        app_repository_module.get_app_repository_info(str(app_dir), False)
    except AppMetadataError as exc:
        assert "Missing keys in app.json" in str(exc)
        assert "description" in str(exc)
    else:
        raise AssertionError("Expected invalid app metadata to raise AppMetadataError")


def test_get_app_repository_info_supports_local_directory(tmp_path: Path) -> None:
    app_dir = tmp_path / "demo_app"
    app_dir.mkdir()
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "Local test app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
            }
        ),
        encoding="utf-8",
    )

    repo = app_repository_module.get_app_repository_info(str(app_dir), False)

    assert isinstance(repo, app_repository_module.LocalAppRepositoryFromDirectory)
    assert repo.get_display_name() == "Demo App"
    assert repo.get_description() == "Local test app"


def test_get_app_repository_info_prefers_windows_local_path_over_hf_identifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "demo_app"
    app_dir.mkdir()
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "Local test app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
            }
        ),
        encoding="utf-8",
    )

    win_path = r"C:\Users\runneradmin\demo_app"

    monkeypatch.setattr(
        app_repository_module, "_resolve_local_app_path", lambda app_id: app_dir if app_id == win_path else None
    )
    monkeypatch.setattr(
        app_repository_module,
        "LocalAppRepositoryFromHF",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Should not resolve Windows local paths as HF repos")
        ),
    )

    repo = app_repository_module.get_app_repository_info(win_path, False)

    assert isinstance(repo, app_repository_module.LocalAppRepositoryFromDirectory)
    assert repo.get_name() == str(app_dir)
