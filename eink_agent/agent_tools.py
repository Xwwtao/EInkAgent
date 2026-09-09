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
            "Search fictional E Ink devices using explicit user constraints."
        ),
        "parameters": DeviceRequirements.model_json_schema(),
    },
}

GET_DEVICE_DETAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "get_device_detail",
        "description": (
            "Get complete details for one fictional E Ink device by ID."
        ),
        "parameters": GetDeviceDetailArguments.model_json_schema(),
    },
}

COMPARE_DEVICES_TOOL = {
    "type": "function",
    "function": {
        "name": "compare_devices",
        "description": (
            "Compare two to five fictional E Ink devices by their IDs."
        ),
        "parameters": CompareDevicesArguments.model_json_schema(),
    },
}

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

def execute_tool(
    name: str,
    arguments: dict[str, object],
) -> Any:
    """Validate and execute one allowlisted tool."""
    registration = _TOOL_REGISTRY.get(name)

    if registration is None:
        raise ValueError(f"Unsupported tool: {name}")

    argument_model, handler = registration

    validated = argument_model.model_validate(arguments)
    validated_arguments = validated.model_dump(exclude_none=True)

    return handler(**validated_arguments)
