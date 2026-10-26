"""Tests for the MCP server (decision 010).

The SDK's in-process client talks to the server over the real MCP protocol
without a subprocess or network; the demo database comes from conftest.
"""

import asyncio
import json
from pathlib import Path
import re

from mcp import Client

import device_repository
from eink_agent.agent_tools import offered_tools
from eink_agent.mcp_server import build_server

SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "eink-shopping" / "SKILL.md"


class FakeKnowledge:
    def search(self, query):
        return [{"chunk_id": "18-ip-rating#1", "title": "防水等级", "section": "IPX8", "text": "可长时间浸泡",
                 "source_url": "https://example.com/ip", "verified_at": "2026-10-13"}]


def _with_client(server, action):
    async def run():
        async with Client(server) as client:
            return await action(client)
    return asyncio.run(run())


def _list_tools(server):
    return _with_client(server, lambda client: client.list_tools()).tools


def _call(server, name, arguments):
    result = _with_client(server, lambda client: client.call_tool(name, arguments))
    return result.is_error, json.loads(result.content[0].text)


def test_device_tools_match_the_agent_and_memory_is_not_exposed():
    tools = {tool.name: tool for tool in _list_tools(build_server())}
    agent_tools = {tool["function"]["name"]: tool["function"] for tool in offered_tools()}

    assert set(tools) == set(agent_tools) == {"search_devices", "get_device_detail", "compare_devices"}
    for name, tool in tools.items():
        assert tool.description == agent_tools[name]["description"]
        assert tool.input_schema == agent_tools[name]["parameters"]
        assert tool.annotations.read_only_hint is True


def test_knowledge_tool_appears_only_with_a_searcher():
    names = [tool.name for tool in _list_tools(build_server(knowledge=FakeKnowledge()))]

    assert "search_knowledge" in names
    assert "remember_preferences" not in names


def test_search_returns_devices_as_json():
    is_error, result = _call(build_server(), "search_devices", {"max_price": 2000})

    assert is_error is False
    assert [device["model"] for device in result] == [
        device["model"] for device in device_repository.search_devices(max_price=2000)
    ]


def test_knowledge_search_returns_passages_with_sources():
    is_error, result = _call(build_server(knowledge=FakeKnowledge()), "search_knowledge", {"query": "IPX8"})

    assert is_error is False
    assert result[0]["source_url"] == "https://example.com/ip"


def test_invalid_arguments_return_a_structured_error():
    is_error, result = _call(build_server(), "search_devices", {"max_price": -5, "colour": True})

    assert is_error is True
    assert result["error"]["type"] == "invalid_arguments"
    assert "max_price" in result["error"]["message"]


def test_unknown_and_memory_tools_are_rejected():
    for name in ("place_order", "remember_preferences"):
        is_error, result = _call(build_server(), name, {})

        assert is_error is True
        assert result["error"]["type"] == "unknown_tool"
        assert result["error"]["available_tools"] == ["search_devices", "get_device_detail", "compare_devices"]


def test_tool_crash_does_not_leak_the_exception(monkeypatch):
    def broken(**_):
        raise RuntimeError("sqlite3 at /Users/secret/path failed")

    from eink_agent import agent_tools
    monkeypatch.setitem(agent_tools._TOOL_REGISTRY, "search_devices", (agent_tools.DeviceRequirements, broken))

    is_error, result = _call(build_server(), "search_devices", {"max_price": 2000})

    assert is_error is True
    assert result["error"]["type"] == "tool_error"
    assert "secret" not in json.dumps(result)


def test_server_instructions_state_the_core_rules():
    instructions = build_server().create_initialization_options().instructions

    assert "read-only" in instructions and "unknown" in instructions


def test_skill_has_metadata_and_names_only_real_tools():
    text = SKILL_PATH.read_text(encoding="utf-8")
    front_matter = re.match(r"^---\n(.*?)\n---\n", text, re.S)

    assert front_matter, "SKILL.md must start with YAML front matter"
    assert re.search(r"^name: eink-shopping$", front_matter.group(1), re.M)
    assert re.search(r"^description: .{20,}", front_matter.group(1), re.M)
    server_tools = {tool.name for tool in _list_tools(build_server(knowledge=FakeKnowledge()))}
    mentioned = set(re.findall(r"`(search_\w+|get_\w+|compare_\w+|remember_\w+)`", text))
    assert mentioned and mentioned <= server_tools
