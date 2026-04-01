import sys
from pathlib import Path

import pytest

import konfai.main as main_module
import konfai.predictor as predictor_module


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
