"""Tool definitions exposed to the language model."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from device_repository import (
    compare_devices,
    get_device_detail,
    search_devices,
)
from eink_agent.requirements import DeviceRequirements

class GetDeviceDetailArguments(BaseModel):
    """Arguments accepted by the device-detail tool."""

    model_config = ConfigDict(extra="forbid")

    device_id: PositiveInt


class CompareDevicesArguments(BaseModel):
    """Arguments accepted by the device-comparison tool."""

    model_config = ConfigDict(extra="forbid")

    device_ids: list[PositiveInt] = Field(
        min_length=2,
        max_length=5,
    )

SEARCH_DEVICES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_devices",
        "description": (
            "Search real E Ink devices using explicit user constraints."
        ),
        "parameters": DeviceRequirements.model_json_schema(),
    },
}

GET_DEVICE_DETAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "get_device_detail",
        "description": (
            "Get complete details for one E Ink device by ID."
        ),
        "parameters": GetDeviceDetailArguments.model_json_schema(),
    },
}

COMPARE_DEVICES_TOOL = {
    "type": "function",
    "function": {
        "name": "compare_devices",
        "description": (
            "Compare two to five E Ink devices by their IDs."
        ),
        "parameters": CompareDevicesArguments.model_json_schema(),
    },
}

AGENT_TOOLS = [
    SEARCH_DEVICES_TOOL,
    GET_DEVICE_DETAIL_TOOL,
    COMPARE_DEVICES_TOOL,
]

_TOOL_REGISTRY = {
    "search_devices": (
        DeviceRequirements,
        search_devices,
    ),
    "get_device_detail": (
        GetDeviceDetailArguments,
        get_device_detail,
    ),
    "compare_devices": (
        CompareDevicesArguments,
        compare_devices,
    ),
}

class UnknownToolError(ValueError):
    """Raised when the model asks for a tool outside the allowlist."""


def tool_names() -> list[str]:
    """Return the allowlisted tool names in the order shown to the model."""
    return [tool["function"]["name"] for tool in AGENT_TOOLS]


def describe_tool_arguments(name: str) -> dict[str, Any]:
    """Summarize a tool's parameters so the model can correct its call."""
    argument_model, _ = _TOOL_REGISTRY[name]
    schema = argument_model.model_json_schema()

    return {
        "required": schema.get("required", []),
        "allowed": sorted(schema.get("properties", {})),
    }


def execute_tool(
    name: str,
    arguments: dict[str, object],
) -> Any:
    """Validate and execute one allowlisted tool."""
    registration = _TOOL_REGISTRY.get(name)

    if registration is None:
        raise UnknownToolError(f"Unsupported tool: {name}")

    argument_model, handler = registration

    validated = argument_model.model_validate(arguments)
    validated_arguments = validated.model_dump(exclude_none=True)

    return handler(**validated_arguments)
