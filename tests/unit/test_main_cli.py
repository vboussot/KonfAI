import sys
from pathlib import Path
from typing import Any, cast

import pytest

import konfai.app as app_module
import konfai.main as main_module
import konfai.predictor as predictor_module


def test_main_apps_dispatches_local_infer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    class DummyApp:
        def __init__(self, app_name: str, download: bool, force_update: bool) -> None:
            calls.append(("init", (app_name, download, force_update)))

        def infer(self, **kwargs) -> None:
            calls.append(("infer", kwargs))

    monkeypatch.setattr(app_module, "KonfAIApp", DummyApp)
    monkeypatch.setattr(app_module, "KonfAIAppClient", object)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "konfai-apps",
            "infer",
            "demo/app",
            "--inputs",
            str(tmp_path / "input.mha"),
            "--cpu",
            "2",
            "--prediction-file",
            "Prediction.custom.yml",
            "--tta",
            "2",
            "--mc",
            "1",
        ],
    )

    main_module.main_apps()

    assert calls[0] == ("init", ("demo/app", False, False))
    infer_kwargs = cast(dict[str, Any], calls[1][1])
    assert infer_kwargs["cpu"] == 2
    assert infer_kwargs["gpu"] == []
    assert infer_kwargs["prediction_file"] == "Prediction.custom.yml"
    assert infer_kwargs["tta"] == 2
    assert infer_kwargs["mc"] == 1
    assert infer_kwargs["inputs"] == [[(tmp_path / "input.mha").resolve()]]


def test_main_prediction_dispatches_config_as_prediction_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_predict(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(predictor_module, "predict", fake_predict)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "konfai",
            "PREDICTION",
            "-c",
            "Prediction.custom.yml",
            "--models",
            str(tmp_path / "checkpoint.pt"),
            "--cpu",
            "1",
        ],
    )

    main_module.main()

    assert captured["prediction_file"] == "Prediction.custom.yml"
    assert captured["cpu"] == 1
    assert captured["gpu"] == []
    assert captured["models"] == [str(tmp_path / "checkpoint.pt")]
    assert "config" not in captured
