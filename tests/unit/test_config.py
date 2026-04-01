from pathlib import Path
from typing import Literal

import pytest

from konfai.utils.config import Config, apply_config, config
from konfai.utils.errors import ConfigError


def _fail_input(_: str) -> str:
    raise AssertionError("input should not be used")


def _configure_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
    *,
    mode: str = "Done",
) -> Path:
    config_path = tmp_path / "config.yml"
    config_path.write_text(content, encoding="utf-8")
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", mode)
    return config_path


def test_config_missing_file_raises_clear_error_without_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "missing.yml"
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    monkeypatch.setattr("builtins.input", _fail_input)

    with pytest.raises(ConfigError, match="does not exist"):
        with Config("Trainer"):
            pass


def test_config_default_mode_materializes_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "generated.yml"
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "default")
    monkeypatch.setattr("builtins.input", _fail_input)

    with Config("Trainer") as config:
        value = config.get_value("train_name", "default|SMOKE")

    assert config_path.exists()
    assert value == "SMOKE"
    content = config_path.read_text(encoding="utf-8")
    assert "Trainer:" in content
    assert "train_name: SMOKE" in content


def test_apply_config_preserves_none_for_optional_nested_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("Root:\n  child: None\n", encoding="utf-8")
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")

    @config("child")
    class Child:
        def __init__(self, value: int = 1) -> None:
            self.value = value

    class Root:
        def __init__(self, child: Child | None = None) -> None:
            self.child = child

    root = apply_config("Root")(Root)()

    assert root.child is None


def test_apply_config_accepts_literal_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(
        tmp_path,
        monkeypatch,
        "Root:\n  mode: eval\n",
    )

    class Root:
        def __init__(self, mode: Literal["train", "eval"] = "train") -> None:
            self.mode = mode

    root = apply_config("Root")(Root)()

    assert root.mode == "eval"


def test_apply_config_rejects_invalid_literal_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(
        tmp_path,
        monkeypatch,
        "Root:\n  mode: invalid\n",
    )

    class Root:
        def __init__(self, mode: Literal["train", "eval"] = "train") -> None:
            self.mode = mode

    with pytest.raises(ConfigError, match="Invalid value 'invalid'"):
        apply_config("Root")(Root)()


def test_apply_config_instantiates_dict_of_nested_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(
        tmp_path,
        monkeypatch,
        ("Root:\n" "  children:\n" "    left:\n" "      value: 3\n" "    right:\n" "      value: 7\n"),
    )

    class Child:
        def __init__(self, value: int) -> None:
            self.value = value

    class Root:
        def __init__(self, children: dict[str, Child]) -> None:
            self.children = children

    root = apply_config("Root")(Root)()

    assert sorted(root.children) == ["left", "right"]
    assert root.children["left"].value == 3
    assert root.children["right"].value == 7


def test_apply_config_preserves_dict_of_primitives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(
        tmp_path,
        monkeypatch,
        ("Root:\n" "  weights:\n" "    mae: 1\n" "    ssim: 2\n"),
    )

    class Root:
        def __init__(self, weights: dict[str, int]) -> None:
            self.weights = weights

    root = apply_config("Root")(Root)()

    assert root.weights == {"mae": 1, "ssim": 2}


def test_apply_config_converts_sequence_of_union_scalars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(
        tmp_path,
        monkeypatch,
        ("Root:\n" "  values:\n" "    - '1'\n" "    - 2\n" "    - '3'\n"),
    )

    class Root:
        def __init__(self, values: list[int | float]) -> None:
            self.values = values

    root = apply_config("Root")(Root)()

    assert root.values == [1, 2, 3]
    assert all(isinstance(value, int) for value in root.values)


def test_apply_config_honors_konfai_without_for_skipped_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(
        tmp_path,
        monkeypatch,
        ("Root:\n" "  kept: 5\n" "  skipped: 42\n"),
    )

    class Root:
        def __init__(self, kept: int, skipped: int = 0) -> None:
            self.kept = kept
            self.skipped = skipped

    root = apply_config("Root")(Root)(konfai_without=["skipped"])

    assert root.kept == 5
    assert root.skipped == 0
