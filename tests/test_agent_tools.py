import json

import pytest
from pydantic import ValidationError

from eink_agent.agent_tools import (
    COMPARE_DEVICES_TOOL,
    GET_DEVICE_DETAIL_TOOL,
    SEARCH_DEVICES_TOOL,
    SEARCH_KNOWLEDGE_TOOL,
    RememberPreferencesArguments,
    UnknownToolError,
    execute_tool,
)
from eink_agent.requirements import DeviceRequirements


def test_search_devices_tool_has_function_schema():
    function = SEARCH_DEVICES_TOOL["function"]

    assert SEARCH_DEVICES_TOOL["type"] == "function"
    assert function["name"] == "search_devices"
    assert function["parameters"]["type"] == "object"
    assert "max_price" in function["parameters"]["properties"]

def test_execute_tool_runs_allowed_search():
    devices = execute_tool(
        "search_devices",
        {"max_price": 2000},
    )

    assert [device["model"] for device in devices] == [
        "Reader 6",
        "Color 7",
    ]


def test_execute_tool_rejects_unknown_tool():
    with pytest.raises(ValueError, match="Unsupported tool"):
        execute_tool("delete_device", {})

def test_execute_tool_rejects_invalid_arguments():
    with pytest.raises(ValidationError):
        execute_tool(
            "search_devices",
            {"max_price": -1},
        )


def test_execute_tool_rejects_unexpected_arguments():
    with pytest.raises(ValidationError):
        execute_tool(
            "search_devices",
            {"delete_everything": True},
        )


def test_device_detail_tool_has_required_positive_id():
    function = GET_DEVICE_DETAIL_TOOL["function"]
    parameters = function["parameters"]

    assert function["name"] == "get_device_detail"
    assert parameters["required"] == ["device_id"]
    assert parameters["properties"]["device_id"]["exclusiveMinimum"] == 0


def test_compare_devices_tool_requires_two_to_five_ids():
    function = COMPARE_DEVICES_TOOL["function"]
    device_ids = function["parameters"]["properties"]["device_ids"]

    assert function["name"] == "compare_devices"
    assert device_ids["minItems"] == 2
    assert device_ids["maxItems"] == 5
    assert device_ids["items"]["exclusiveMinimum"] == 0

def test_execute_tool_returns_device_detail():
    device = execute_tool(
        "get_device_detail",
        {"device_id": 1},
    )

    assert device is not None
    assert device["model"] == "Reader 6"


def test_execute_tool_compares_devices_in_requested_order():
    devices = execute_tool(
        "compare_devices",
        {"device_ids": [3, 1]},
    )

    assert [device["model"] for device in devices] == [
        "Color 7",
        "Reader 6",
    ]


def test_execute_tool_rejects_invalid_detail_id():
    with pytest.raises(ValidationError):
        execute_tool(
            "get_device_detail",
            {"device_id": 0},
        )


def test_execute_tool_rejects_too_few_comparison_ids():
    with pytest.raises(ValidationError):
        execute_tool(
            "compare_devices",
            {"device_ids": [1]},
        )


def test_search_category_only_accepts_known_values():
    with pytest.raises(ValidationError):
        execute_tool("search_devices", {"category": "阅读器"})


def test_search_tool_schema_lists_category_values():
    category = SEARCH_DEVICES_TOOL["function"]["parameters"]["properties"]["category"]

    assert {"reader", "note", "tablet"} <= set(json.dumps(category).replace('"', " ").split())


def test_knowledge_tool_asks_for_chinese_queries():
    # d01 (evals/reports/20261013T143639Z): an English query missed the
    # Chinese water-resistance document because its score fell below the gate.
    description = SEARCH_KNOWLEDGE_TOOL["function"]["description"]
    assert "Chinese" in description


class FakeMemory:
    def __init__(self):
        self.saved = []
        self.current = {}

    def remember(self, updates):
        self.saved.append(updates)
        self.current = {**self.current, **updates}
        self.current = {k: v for k, v in self.current.items() if v is not None}
        return self.current


def test_remember_preferences_saves_only_the_fields_given():
    memory = FakeMemory()

    result = execute_tool("remember_preferences", {"max_price": 1500, "supports_stylus": True}, memory=memory)

    assert memory.saved == [{"max_price": 1500, "supports_stylus": True}]
    assert result == {"saved": {"max_price": 1500, "supports_stylus": True},
                      "current": {"max_price": 1500, "supports_stylus": True}}


def test_remember_preferences_passes_explicit_null_to_withdraw():
    memory = FakeMemory()
    memory.current = {"supports_stylus": True}

    result = execute_tool("remember_preferences", {"supports_stylus": None}, memory=memory)

    assert memory.saved == [{"supports_stylus": None}]
    assert result["current"] == {}


@pytest.mark.parametrize(
    "arguments",
    [{}, {"max_price": -1}, {"category": "阅读器"}, {"keyword": "Kindle"}, {"brand": ""}],
)
def test_remember_preferences_rejects_invalid_arguments_and_saves_nothing(arguments):
    memory = FakeMemory()

    with pytest.raises(ValidationError):
        execute_tool("remember_preferences", arguments, memory=memory)

    assert memory.saved == []


def test_remember_preferences_does_not_exist_without_memory():
    with pytest.raises(UnknownToolError):
        execute_tool("remember_preferences", {"max_price": 1500})


def test_preference_fields_are_search_fields():
    # Remembered values must be usable as search_devices arguments.
    assert set(RememberPreferencesArguments.model_fields) <= set(DeviceRequirements.model_fields)
