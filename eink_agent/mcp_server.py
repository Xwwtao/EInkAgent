"""The device tools as an MCP server, for clients such as Claude Desktop
and Claude Code (decision 010).

    python -m eink_agent.mcp_server

Runs over stdio, so nothing here may print to standard output. Tool names,
descriptions, argument schemas, validation and error types all come from
the code the Agent uses; this module only translates MCP requests.
"""

from functools import partial
import json
from typing import Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from eink_agent.agent import ToolCallError, run_tool
from eink_agent.agent_tools import offered_tools

INSTRUCTIONS = (
    "Tools over a database of real E Ink devices and price snapshots. All "
    "tools are read-only: nothing can be ordered, paid or changed. A null "
    "field means the source did not publish it: call it unknown and do not "
    "infer it. Give each price with its seller and collection date. "
    "search_knowledge, when present, returns cited passages or an empty "
    "list; when it is empty, say no supporting material was found."
)

READ_ONLY = types.ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


def _text_result(payload: Any, *, is_error: bool) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        is_error=is_error,
    )


def build_server(knowledge: Any = None) -> Server:
    """An MCP server over the Agent's tools.

    ``knowledge`` adds search_knowledge, as for the Agent. The memory tool
    is never offered: it belongs to the web page's sessions.
    """
    tools = [
        types.Tool(
            name=tool["function"]["name"],
            description=tool["function"]["description"],
            input_schema=tool["function"]["parameters"],
            annotations=READ_ONLY,
        )
        for tool in offered_tools(include_knowledge=knowledge is not None)
    ]

    async def list_tools(_context: Any, _params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def call_tool(_context: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        try:
            # Tools use blocking SQLite and HTTP calls, so they run in a
            # worker thread instead of stalling the protocol loop.
            result = await anyio.to_thread.run_sync(
                partial(run_tool, params.name, params.arguments or {}, knowledge)
            )
        except ToolCallError as error:
            return _text_result({"error": error.payload}, is_error=True)
        return _text_result(result, is_error=False)

    return Server(
        "eink-agent",
        instructions=INSTRUCTIONS,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def _serve(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    # Imported here so the tests can build a server without the web app.
    from eink_agent.api import _knowledge_searcher

    anyio.run(_serve, build_server(knowledge=_knowledge_searcher()))


if __name__ == "__main__":
    main()
