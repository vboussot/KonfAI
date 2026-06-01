import json
from pathlib import Path

import pytest
from huggingface_hub import constants as hf_constants
from huggingface_hub import file_download as hf_file_download
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


def test_local_hf_get_filenames_returns_relative_files_and_ignores_folders(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFolder:
        def __init__(self, path: str) -> None:
            self.path = path

    class DummyFile:
        def __init__(self, path: str) -> None:
            self.path = path

    monkeypatch.setattr(app_repository_module, "RepoFolder", DummyFolder)
    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "_list_repo_tree",
        staticmethod(
            lambda repo_id, app_name, recursive=False: [
                DummyFolder("demo_app/assets"),
                DummyFile("demo_app/app.json"),
                DummyFile("demo_app/Inference.yml"),
                DummyFile("demo_app/assets/preprocess.py"),
            ]
        ),
    )

    filenames = app_repository_module.LocalAppRepositoryFromHF.get_filenames("org/demo", "demo_app", True)

    assert filenames == ["Inference.yml", "app.json", "assets/preprocess.py"]


def test_is_app_repo_requires_root_app_json() -> None:
    assert app_repository_module.is_app_repo(["Inference.yml", "app.json", "weights/model.pt"])
    assert not app_repository_module.is_app_repo(["docs/app.json", "Inference.yml"])


def test_local_hf_download_syncs_non_model_files_for_current_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class DummyFolder:
        def __init__(self, path: str) -> None:
            self.path = path

    class DummyFile:
        def __init__(self, path: str) -> None:
            self.path = path

    cache_dir = tmp_path / "hf-cache"
    lock_dir = cache_dir / ".locks" / "repo-lock"
    lock_dir.mkdir(parents=True)
    snapshot_dir = tmp_path / "snapshot"
    (snapshot_dir / "demo_app").mkdir(parents=True)

    calls: dict[str, object] = {}

    monkeypatch.setattr(app_repository_module, "RepoFolder", DummyFolder)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cache_dir))
    monkeypatch.setattr(hf_file_download, "repo_folder_name", lambda repo_id, repo_type: "repo-lock")
    monkeypatch.setattr(app_repository_module.shutil, "rmtree", lambda path: calls.setdefault("removed", str(path)))
    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "_list_repo_tree",
        staticmethod(
            lambda repo_id, app_name, recursive=False: [
                DummyFile("demo_app/app.json"),
                DummyFile("demo_app/Inference.yml"),
                DummyFolder("demo_app/assets"),
                DummyFile("demo_app/assets/preprocess.py"),
                DummyFile("demo_app/model.pt"),
            ]
        ),
    )

    def fake_snapshot_download(**kwargs):
        calls["snapshot"] = kwargs
        return str(snapshot_dir)

    monkeypatch.setattr(app_repository_module, "snapshot_download", fake_snapshot_download)

    result = app_repository_module.LocalAppRepositoryFromHF.download(
        "org/demo@refs/pr/1",
        "demo_app/Inference.yml",
        True,
    )

    assert result == snapshot_dir / "demo_app" / "Inference.yml"
    assert calls["removed"] == str(lock_dir)
    assert calls["snapshot"] == {
        "repo_id": "org/demo",
        "repo_type": "model",
        "revision": "refs/pr/1",
        "allow_patterns": [
            "demo_app/Inference.yml",
            "demo_app/app.json",
            "demo_app/assets/preprocess.py",
        ],
    }


def test_local_hf_initial_model_availability_comes_from_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_json = tmp_path / "app.json"
    app_json.write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "HF test app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["model.pt"],
            }
        ),
        encoding="utf-8",
    )
    inference_yml = tmp_path / "Inference.yml"
    inference_yml.write_text("Predictor: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "get_filenames",
        staticmethod(lambda repo_id, app_name, force_update: ["Inference.yml", "app.json", "model.pt"]),
    )
    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "get_cached_filenames",
        staticmethod(lambda repo_id, app_name: ["Inference.yml", "app.json"]),
    )

    def fake_download(repo_id: str, filename: str, force_update: bool) -> Path:
        if filename.endswith("app.json"):
            return app_json
        if filename.endswith("Inference.yml"):
            return inference_yml
        return tmp_path / Path(filename).name

    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "download",
        staticmethod(fake_download),
    )

    repo = app_repository_module.LocalAppRepositoryFromHF("org/demo", "demo_app", True)

    assert repo.get_checkpoints_name() == ["model.pt"]
    assert repo.get_checkpoints_name_available() == []


def test_local_hf_nested_cached_model_is_reported_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_json = tmp_path / "app.json"
    app_json.write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "HF test app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["model.pt"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "get_filenames",
        staticmethod(lambda repo_id, app_name, force_update: ["Inference.yml", "app.json", "weights/model.pt"]),
    )
    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "get_cached_filenames",
        staticmethod(lambda repo_id, app_name: ["Inference.yml", "app.json", "weights/model.pt"]),
    )
    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "download",
        staticmethod(lambda repo_id, filename, force_update: app_json),
    )

    repo = app_repository_module.LocalAppRepositoryFromHF("org/demo", "demo_app", True)

    assert repo.get_checkpoints_name_available() == ["model.pt"]


def test_local_hf_download_inference_refreshes_selected_remote_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_json = tmp_path / "app.json"
    app_json.write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "HF test app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["CV_0.pt"],
            }
        ),
        encoding="utf-8",
    )
    inference_yml = tmp_path / "Inference.yml"
    inference_yml.write_text("Predictor: {}\n", encoding="utf-8")

    get_filenames_calls: list[bool] = []

    def fake_get_filenames(repo_id: str, app_name: str, force_update: bool) -> list[str]:
        get_filenames_calls.append(force_update)
        if force_update:
            return ["CV_0.pt", "Inference.yml", "app.json"]
        return ["Inference.yml", "app.json"]

    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "get_filenames",
        staticmethod(fake_get_filenames),
    )
    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "get_cached_filenames",
        staticmethod(lambda repo_id, app_name: ["Inference.yml", "app.json"]),
    )

    def fake_download(repo_id: str, filename: str, force_update: bool) -> Path:
        if filename.endswith("app.json"):
            return app_json
        if filename.endswith("Inference.yml"):
            return inference_yml
        return tmp_path / Path(filename).name

    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "download",
        staticmethod(fake_download),
    )

    repo = app_repository_module.LocalAppRepositoryFromHF("org/demo", "demo_app", False)

    models_path, prediction_path, codes_path = repo.download_inference(1, ["CV_0"], "Inference.yml")

    assert models_path == [tmp_path / "CV_0.pt"]
    assert prediction_path == inference_yml
    assert codes_path == []
    assert get_filenames_calls == [False, False, True]


def test_local_directory_install_inference_preserves_nested_python_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    (app_root / "pkg").mkdir(parents=True)
    (app_root / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "Local nested app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["model.pt"],
            }
        ),
        encoding="utf-8",
    )
    (app_root / "Inference.yml").write_text(
        "Predictor:\n  Dataset:\n    augmentations: {}\n    Patch:\n      patch_size: [1, 1, 1]\n    batch_size: 1\n",
        encoding="utf-8",
    )
    (app_root / "model.pt").write_text("weights", encoding="utf-8")
    (app_root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (app_root / "pkg" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    repo.install_inference(
        number_of_augmentation=0,
        number_of_model=1,
        name_of_models=[],
        number_of_mc_dropout=0,
        uncertainty=True,
        prediction_file="Inference.yml",
        available_vram=None,
    )

    assert (workspace / "pkg" / "__init__.py").exists()
    assert (workspace / "pkg" / "helper.py").exists()


def test_local_directory_nested_uncertainty_file_is_detected(tmp_path: Path) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    (app_root / "qa").mkdir(parents=True)
    (app_root / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "Local nested app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
            }
        ),
        encoding="utf-8",
    )
    (app_root / "qa" / "Uncertainty.yml").write_text("Predictor: {}\n", encoding="utf-8")

    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)

    assert repo.has_capabilities() == (False, False, True)
