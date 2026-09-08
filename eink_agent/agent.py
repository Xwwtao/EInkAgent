"""Minimal tool-calling agent loop."""

import json
from typing import Any

from openai import OpenAI

from eink_agent.agent_tools import SEARCH_DEVICES_TOOL, execute_tool


SYSTEM_PROMPT = (
    "You are EInkAgent, an assistant for comparing fictional demo E Ink "
    "devices. Use the available search tool when device data is needed. "
    "Base device claims only on tool results. Clearly state that results "
    "are fictional demo data."
)


def run_agent(
    user_text: str,
    *,
    client: OpenAI,
    model: str,
    max_rounds: int = 3,
) -> str:
    """Run a bounded model-tool-model conversation."""
    normalized_text = user_text.strip()
    if not normalized_text:
        raise ValueError("user_text must not be empty")

    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": normalized_text},
    ]

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[SEARCH_DEVICES_TOOL],
            tool_choice="auto",
        )

        if not response.choices:
            raise RuntimeError("Model did not return a response")

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            content = message.content
            if not content or not content.strip():
                raise RuntimeError("Model did not return a final answer")
            return content

        messages.append(message)

        for tool_call in tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            result = execute_tool(
                tool_call.function.name,
                arguments,
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    raise RuntimeError("Agent exceeded maximum rounds")