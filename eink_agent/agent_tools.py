"""Tool definitions exposed to the language model."""

from typing import Any

from device_repository import search_devices
from eink_agent.requirements import DeviceRequirements

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

_TOOL_HANDLERS = {
    "search_devices": search_devices,
}


def execute_tool(
    name: str,
    arguments: dict[str, object],
) -> list[dict[str, Any]]:
    """Validate and execute one allowlisted tool."""
    handler = _TOOL_HANDLERS.get(name)

    if handler is None:
        raise ValueError(f"Unsupported tool: {name}")

    requirements = DeviceRequirements.model_validate(arguments)
    validated_arguments = requirements.model_dump(exclude_none=True)

    return handler(**validated_arguments)
