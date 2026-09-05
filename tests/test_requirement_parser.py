"""Tests for LLM-backed requirement parsing."""

from unittest.mock import Mock

import pytest

from eink_agent.requirement_parser import parse_requirements
from eink_agent.requirements import DeviceRequirements

from pydantic import ValidationError


USER_TEXT = "我想买一台 2000 元以内、支持手写的设备"


def test_parse_requirements_returns_structured_output():
    client = Mock()
    expected = DeviceRequirements(
        max_price=2000,
        supports_stylus=True,
    )

    client.chat.completions.create.return_value.choices = [
        Mock(
            finish_reason="stop",
            message=Mock(content=expected.model_dump_json()),
        )
    ]

    result = parse_requirements(
        USER_TEXT,
        client=client,
        model="test-model",
    )

    assert result == expected

    client.chat.completions.create.assert_called_once()
    request = client.chat.completions.create.call_args.kwargs

    assert request["model"] == "test-model"
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][-1] == {
        "role": "user",
        "content": USER_TEXT,
    }


def test_parse_requirements_rejects_empty_input():
    client = Mock()

    with pytest.raises(ValueError, match="must not be empty"):
        parse_requirements(
            "   ",
            client=client,
            model="test-model",
        )

    client.chat.completions.create.assert_not_called()


def test_parse_requirements_rejects_missing_structured_output():
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        Mock(finish_reason="stop", message=Mock(content=None))
    ]

    with pytest.raises(RuntimeError, match="structured requirements"):
        parse_requirements(
            USER_TEXT,
            client=client,
            model="test-model",
        )

def test_parse_requirements_rejects_invalid_model_values():
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        Mock(
            finish_reason="stop",
            message=Mock(content='{"max_price": -200}'),
        )
    ]

    with pytest.raises(ValidationError):
        parse_requirements(
            USER_TEXT,
            client=client,
            model="test-model",
        )