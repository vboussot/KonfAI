# SPDX-License-Identifier: Apache-2.0
"""Tests verifying that CLI subcommands dispatch to the correct backend functions."""

import sys

import konfai.evaluator as evaluator_module
import konfai.main as main_module
import konfai.trainer as trainer_module
import pytest


def test_konfai_help_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["konfai", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 0


def test_konfai_train_dispatches_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_train(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(trainer_module, "train", fake_train)
    monkeypatch.setattr(sys, "argv", ["konfai", "TRAIN", "-c", "Config.yml"])

    main_module.main()

    assert captured["config"] == "Config.yml"


def test_konfai_eval_dispatches_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(evaluator_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(sys, "argv", ["konfai", "EVALUATION", "-c", "Evaluation.yml"])

    main_module.main()

    assert captured["evaluations_file"] == "Evaluation.yml"
