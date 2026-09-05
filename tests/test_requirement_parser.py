"""Tests for LLM-backed requirement parsing."""

from unittest.mock import Mock

import pytest

from eink_agent.requirement_parser import parse_requirements
from eink_agent.requirements import DeviceRequirements


USER_TEXT = "我想买一台 2000 元以内、支持手写的设备"


def test_parse_requirements_returns_structured_output():
    client = Mock()
    expected = DeviceRequirements(
        max_price=2000,
        supports_stylus=True,
    )
    client.responses.parse.return_value.output_parsed = expected

    result = parse_requirements(
        USER_TEXT,
        client=client,
        model="test-model",
    )

    assert result == expected

    request = client.responses.parse.call_args.kwargs
    assert request["model"] == "test-model"
    assert request["text_format"] is DeviceRequirements
    assert request["input"][-1] == {
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

    client.responses.parse.assert_not_called()


def test_parse_requirements_rejects_missing_structured_output():
    client = Mock()
    client.responses.parse.return_value.output_parsed = None

    with pytest.raises(RuntimeError, match="structured requirements"):
        parse_requirements(
            USER_TEXT,
            client=client,
            model="test-model",
        )