import pytest

from eink_agent.agent_tools import (
    COMPARE_DEVICES_TOOL,
    GET_DEVICE_DETAIL_TOOL,
    SEARCH_DEVICES_TOOL,
    execute_tool,
)

from pydantic import ValidationError


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