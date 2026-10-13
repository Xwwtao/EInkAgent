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

class SearchKnowledgeArguments(BaseModel):
    """Arguments accepted by the knowledge-search tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=200,
        description="The user's question about E Ink technology, in plain words.",
    )


SEARCH_KNOWLEDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": (
            "Search short explanations of E Ink technology: screen panels "
            "(Carta, Kaleido, Gallery), front and warm light, styluses, open "
            "and closed systems, file formats, water resistance, eye-comfort "
            "claims and battery life. Returns up to 3 passages with sources, "
            "or an empty list when nothing relevant exists. Not for device "
            "specs or prices. The passages are written in Chinese, so write "
            "the query in Chinese."
        ),
        "parameters": SearchKnowledgeArguments.model_json_schema(),
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

class UnknownToolError(ValueError):
    """Raised when the model asks for a tool outside the allowlist."""


def tool_names(include_knowledge: bool = False) -> list[str]:
    """Return the allowlisted tool names in the order shown to the model."""
    tools = AGENT_TOOLS + ([SEARCH_KNOWLEDGE_TOOL] if include_knowledge else [])
    return [tool["function"]["name"] for tool in tools]


def describe_tool_arguments(name: str) -> dict[str, Any]:
    """Summarize a tool's parameters so the model can correct its call."""
    argument_model = (
        SearchKnowledgeArguments if name == "search_knowledge" else _TOOL_REGISTRY[name][0]
    )
    schema = argument_model.model_json_schema()

    return {
        "required": schema.get("required", []),
        "allowed": sorted(schema.get("properties", {})),
    }


def execute_tool(
    name: str,
    arguments: dict[str, object],
    knowledge: Any = None,
) -> Any:
    """Validate and execute one allowlisted tool.

    ``knowledge`` is the searcher behind search_knowledge; without it the
    tool does not exist.
    """
    if name == "search_knowledge" and knowledge is not None:
        validated = SearchKnowledgeArguments.model_validate(arguments)
        return knowledge.search(validated.query)

    registration = _TOOL_REGISTRY.get(name)

    if registration is None:
        raise UnknownToolError(f"Unsupported tool: {name}")

    argument_model, handler = registration

    validated = argument_model.model_validate(arguments)
    validated_arguments = validated.model_dump(exclude_none=True)

    return handler(**validated_arguments)
