# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for improved ConfigError messages introduced in Phase 05."""

from pathlib import Path
from typing import Literal

import pytest
from konfai.utils.config import Config, apply_config
from konfai.utils.errors import ConfigError


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


def test_missing_config_file_error_contains_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "missing.yml"
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")

    with pytest.raises(ConfigError) as exc_info:
        with Config("Trainer"):
            pass

    msg = str(exc_info.value)
    assert "missing.yml" in msg
    assert "does not exist" in msg


def test_missing_config_file_error_contains_mode_and_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "missing.yml"
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")

    with pytest.raises(ConfigError) as exc_info:
        with Config("Trainer"):
            pass

    msg = str(exc_info.value)
    assert "KONFAI_CONFIG_MODE=Done" in msg
    assert "konfai TRAINING" in msg


def test_invalid_yaml_syntax_raises_config_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "broken.yml"
    config_path.write_text("key: {unclosed\n", encoding="utf-8")
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")

    with pytest.raises(ConfigError) as exc_info:
        with Config("Root"):
            pass

    msg = str(exc_info.value)
    assert "Invalid YAML syntax" in msg
    assert "broken.yml" in msg


def test_type_mismatch_error_names_field_and_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(tmp_path, monkeypatch, "Root:\n  count: hello\n")

    class Root:
        def __init__(self, count: int = 0) -> None:
            self.count = count

    with pytest.raises(ConfigError) as exc_info:
        apply_config("Root")(Root)()

    msg = str(exc_info.value)
    assert "count" in msg
    assert "int" in msg


@pytest.mark.parametrize(
    ("literal", "expected"),
    [("true", True), ("1", True), ("yes", True), ("false", False), ("0", False), ("no", False)],
)
def test_apply_config_parses_boolean_strings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    literal: str,
    expected: bool,
) -> None:
    _configure_env(tmp_path, monkeypatch, f"Root:\n  enabled: '{literal}'\n")

    class Root:
        def __init__(self, enabled: bool = True) -> None:
            self.enabled = enabled

    assert apply_config("Root")(Root)().enabled is expected


def test_apply_config_rejects_unknown_boolean_string(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(tmp_path, monkeypatch, "Root:\n  enabled: 'sometimes'\n")

    class Root:
        def __init__(self, enabled: bool = True) -> None:
            self.enabled = enabled

    with pytest.raises(ConfigError, match="expected bool"):
        apply_config("Root")(Root)()


def test_invalid_literal_raises_config_error_with_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(tmp_path, monkeypatch, "Root:\n  mode: invalid\n")

    class Root:
        def __init__(self, mode: Literal["train", "eval"] = "train") -> None:
            self.mode = mode

    with pytest.raises(ConfigError) as exc_info:
        apply_config("Root")(Root)()

    msg = str(exc_info.value)
    assert "invalid" in msg
    # The error must mention the valid options
    assert "train" in msg or "eval" in msg
