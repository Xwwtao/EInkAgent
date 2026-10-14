"""Tool definitions exposed to the language model."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

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

class RememberPreferencesArguments(BaseModel):
    """Lasting requirements the user stated or changed (decision 009).

    Constraints match search_devices, so remembered values can be passed to
    it unchanged. A field sent as null withdraws that preference.
    """

    model_config = ConfigDict(extra="forbid")

    max_price: int | None = Field(default=None, ge=0)
    min_screen_size: float | None = Field(default=None, gt=0)
    max_screen_size: float | None = Field(default=None, gt=0)
    max_weight_g: int | None = Field(default=None, gt=0)
    supports_stylus: bool | None = None
    is_color: bool | None = None
    is_open_system: bool | None = None
    category: Literal["reader", "note", "tablet"] | None = None
    brand: str | None = Field(default=None, min_length=1, max_length=40)

    @model_validator(mode="after")
    def _at_least_one_field(self) -> "RememberPreferencesArguments":
        if not self.model_fields_set:
            raise ValueError("give at least one preference to remember")
        return self


REMEMBER_PREFERENCES_TOOL = {
    "type": "function",
    "function": {
        "name": "remember_preferences",
        "description": (
            "Save lasting requirements the user stated or changed in this "
            "conversation (budget, screen size, weight, stylus, colour, open "
            "system, device type, brand). Send only the fields that changed; "
            "send null for a field the user no longer wants. Returns the "
            "preferences now in effect."
        ),
        "parameters": RememberPreferencesArguments.model_json_schema(),
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


def offered_tools(include_knowledge: bool = False, include_memory: bool = False) -> list[dict[str, Any]]:
    """Tools shown to the model; the optional ones need their backend."""
    return (
        AGENT_TOOLS
        + ([SEARCH_KNOWLEDGE_TOOL] if include_knowledge else [])
        + ([REMEMBER_PREFERENCES_TOOL] if include_memory else [])
    )


def tool_names(include_knowledge: bool = False, include_memory: bool = False) -> list[str]:
    """Return the allowlisted tool names in the order shown to the model."""
    return [tool["function"]["name"] for tool in offered_tools(include_knowledge, include_memory)]


def describe_tool_arguments(name: str) -> dict[str, Any]:
    """Summarize a tool's parameters so the model can correct its call."""
    optional = {"search_knowledge": SearchKnowledgeArguments, "remember_preferences": RememberPreferencesArguments}
    argument_model = optional[name] if name in optional else _TOOL_REGISTRY[name][0]
    schema = argument_model.model_json_schema()

    return {
        "required": schema.get("required", []),
        "allowed": sorted(schema.get("properties", {})),
    }


def execute_tool(
    name: str,
    arguments: dict[str, object],
    knowledge: Any = None,
    memory: Any = None,
) -> Any:
    """Validate and execute one allowlisted tool.

    ``knowledge`` is the searcher behind search_knowledge and ``memory`` the
    session memory behind remember_preferences; without them those tools do
    not exist.
    """
    if name == "search_knowledge" and knowledge is not None:
        validated = SearchKnowledgeArguments.model_validate(arguments)
        return knowledge.search(validated.query)

    if name == "remember_preferences" and memory is not None:
        # exclude_unset keeps an explicit null, which withdraws a preference.
        updates = RememberPreferencesArguments.model_validate(arguments).model_dump(exclude_unset=True)
        return {"saved": updates, "current": memory.remember(updates)}

    registration = _TOOL_REGISTRY.get(name)

    if registration is None:
        raise UnknownToolError(f"Unsupported tool: {name}")

    argument_model, handler = registration

    validated = argument_model.model_validate(arguments)
    validated_arguments = validated.model_dump(exclude_none=True)

    return handler(**validated_arguments)
