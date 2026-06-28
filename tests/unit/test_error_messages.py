# SPDX-License-Identifier: Apache-2.0
"""Tests verifying that KonfAI error types format their messages correctly."""

from konfai.utils.errors import ConfigError, KonfAIError


def test_named_error_formats_with_type_prefix() -> None:
    error = ConfigError("bad value")

    assert "[Config]" in str(error)
    assert "bad value" in str(error)


def test_named_error_with_multiple_messages_uses_arrow() -> None:
    error = ConfigError("bad value", "expected int", "got str")

    assert "→" in str(error)


def test_konfai_error_without_args_returns_empty_bracket() -> None:
    error = KonfAIError()

    result = str(error)
    assert "[Error]" in result
    assert result
