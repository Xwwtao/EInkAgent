import pytest

from eink_agent.agent_tools import SEARCH_DEVICES_TOOL, execute_tool
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
