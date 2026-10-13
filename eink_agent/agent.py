"""Bounded tool-calling agent loop with structured tool failures.

Design notes: docs/decisions/004-agent-tool-failures.md.
"""

from dataclasses import dataclass, field
import json
import time
from typing import Any, Literal

from openai import OpenAI
from pydantic import ValidationError

from eink_agent.agent_tools import (
    AGENT_TOOLS,
    UnknownToolError,
    describe_tool_arguments,
    execute_tool,
    tool_names,
)
from eink_agent.answer_guard import (
    find_internal_field_names,
    find_unexecuted_action_claims,
    has_leaked_tool_markup,
)

# One failure gives the model a chance to correct itself; a second failure in
# a row means it is not converging, and each extra round costs a model call.
MAX_CONSECUTIVE_TOOL_FAILURES = 2

# About five conversation turns. Every extra message costs tokens and latency,
# and device-buying conversations rarely need more context than this.
MAX_HISTORY_MESSAGES = 10

# Clients may replay what the user and the assistant said, but never system
# instructions or tool results, which they could forge.
HISTORY_ROLES = ("user", "assistant")

# Validation messages beyond this are dropped to keep tool messages short.
MAX_REPORTED_VALIDATION_ERRORS = 5

AgentStatus = Literal["ok", "degraded", "blocked"]

DEGRADED_ANSWER = (
    "抱歉，这次没能完成设备查询。请换个说法，"
    "或者把预算、屏幕尺寸、是否需要手写等条件说得更具体一些。"
)

BLOCKED_ANSWER = (
    "抱歉，我无法确认这个回答的准确性。EInkAgent 只能查询和比较设备，"
    "不能下单、付款或修改任何数据。"
)

CLAIM_REWRITE_INSTRUCTION = (
    "Your answer claims an action that was never executed ({claims}). "
    "EInkAgent cannot place orders, pay, or modify data. Rewrite the answer "
    "using only the tool results, and state the limitation if relevant."
)

FIELD_NAME_REWRITE_INSTRUCTION = (
    "Your answer shows internal field names ({fields}). Rewrite it in plain "
    "language for the user, e.g. say 'closed system' instead of "
    "is_open_system=0, and keep every fact unchanged."
)

LAST_ROUND_INSTRUCTION = (
    "There are no more tool calls available. Answer the user now using only "
    "the tool results above; if nothing matched, say so plainly."
)

SYSTEM_PROMPT = (
    "You are EInkAgent, an assistant that helps users choose E Ink devices "
    "from a database of real devices and price snapshots. Use the available "
    "tools whenever device data is needed. Use search_devices for "
    "constraints, get_device_detail for one known device ID, and "
    "compare_devices for multiple device IDs. "
    "You cannot place orders, purchase products, or modify or delete database "
    "records. For unsupported actions, do not call any tool; clearly explain "
    "the limitation. If a request lacks sufficient constraints or device IDs, "
    "ask a clarifying question without calling tools. "
    "Base device claims only on tool results. A null field means the source "
    "did not publish it: say it is unknown and never infer it. Never show "
    "field names, raw 0/1 flags or null to the user; describe them in plain "
    "language. Prices are lowest-price snapshots from the seller and date "
    "returned with each device; overseas "
    "prices are converted from USD at a fixed rate and are reference prices "
    "only. If a tool returns an error, correct the arguments and try again, "
    "or explain the problem to the user. Answer in the user's language; "
    "reply in Simplified Chinese when the user writes Simplified Chinese."
)


@dataclass
class AgentResult:
    """Final answer, every attempted tool call, and how the run ended.

    ``model_calls`` records latency and token usage for every model request,
    for evaluation reports (AGENTS.md section 7).
    """

    answer: str
    tool_trace: list[dict[str, Any]]
    status: AgentStatus = "ok"
    model_calls: list[dict[str, Any]] = field(default_factory=list)


class ToolCallError(Exception):
    """A tool call failure that is reported back to the model."""

    def __init__(self, error_type: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.payload = {"type": error_type, "message": message, **details}


def _parse_arguments(name: str, raw_arguments: str) -> dict[str, Any]:
    """Decode the model's JSON arguments into an object."""
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError) as error:
        raise ToolCallError(
            "invalid_json",
            "Arguments must be a valid JSON object.",
            **_expected_arguments(name),
        ) from error

    if not isinstance(arguments, dict):
        raise ToolCallError(
            "invalid_json",
            "Arguments must be a JSON object, not a list or a value.",
            **_expected_arguments(name),
        )

    return arguments


def _expected_arguments(name: str) -> dict[str, Any]:
    """Return the argument summary for a known tool, or nothing."""
    if name not in tool_names():
        return {}

    return {"expected_arguments": describe_tool_arguments(name)}


def _describe_validation_error(error: ValidationError) -> str:
    """Summarize which fields failed validation, without echoing inputs."""
    problems = [
        f"{'.'.join(str(part) for part in item['loc']) or 'arguments'}: "
        f"{item['msg']}"
        for item in error.errors()[:MAX_REPORTED_VALIDATION_ERRORS]
    ]

    return "; ".join(problems)


def _execute(name: str, arguments: dict[str, Any]) -> Any:
    """Run one tool, converting every failure into a ToolCallError."""
    try:
        return execute_tool(name, arguments)
    except UnknownToolError as error:
        raise ToolCallError(
            "unknown_tool",
            f"Tool {name!r} does not exist.",
            available_tools=tool_names(),
        ) from error
    except ValidationError as error:
        raise ToolCallError(
            "invalid_arguments",
            _describe_validation_error(error),
            **_expected_arguments(name),
        ) from error
    except Exception as error:
        # The original exception may contain paths, SQL or user input, so the
        # model only learns that the tool failed.
        raise ToolCallError(
            "tool_error",
            "The tool failed while running. Try different arguments or "
            "explain the problem to the user.",
        ) from error


USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    # DeepSeek reports how much of the prompt hit its context cache; other
    # providers may not, in which case these stay None.
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
)


def _record_usage(response: Any, model: str, latency_ms: int) -> dict[str, Any]:
    """Read token usage from a response; missing figures stay None."""
    usage = getattr(response, "usage", None)
    record: dict[str, Any] = {"model": model, "latency_ms": latency_ms}

    for name in USAGE_FIELDS:
        value = getattr(usage, name, None) if usage is not None else None
        record[name] = value if isinstance(value, int) else None

    return record


def _count_results(result: Any) -> int:
    if result is None:
        return 0
    if isinstance(result, list):
        return len(result)
    return 1


def _run_tool_call(tool_call: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one tool call and return its trace entry and tool message."""
    name = tool_call.function.name
    entry: dict[str, Any] = {
        "tool_call_id": tool_call.id,
        "name": name,
        "arguments": None,
    }

    try:
        arguments = _parse_arguments(name, tool_call.function.arguments)
        entry["arguments"] = arguments
        result = _execute(name, arguments)
    except ToolCallError as error:
        entry.update(
            status="error",
            error=error.payload,
            result=None,
            result_count=0,
        )
        content = {"error": error.payload}
    else:
        entry.update(
            status="ok",
            result=result,
            result_count=_count_results(result),
        )
        content = result

    message = {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(content, ensure_ascii=False),
    }

    return entry, message


def _rewrite_instruction(claims: list[str], field_names: list[str]) -> str:
    """Explain every problem found in an answer in one rewrite request."""
    parts = []
    if claims:
        parts.append(CLAIM_REWRITE_INSTRUCTION.format(claims="、".join(claims)))
    if field_names:
        parts.append(FIELD_NAME_REWRITE_INSTRUCTION.format(fields=", ".join(field_names)))
    return " ".join(parts)


def _recent_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Validate earlier turns and keep only the most recent ones."""
    for message in history:
        if message.get("role") not in HISTORY_ROLES:
            raise ValueError(f"history role must be one of {HISTORY_ROLES}")
        if not str(message.get("content", "")).strip():
            raise ValueError("history content must not be empty")

    return [
        {"role": message["role"], "content": message["content"]}
        for message in history[-MAX_HISTORY_MESSAGES:]
    ]


def run_agent(
    user_text: str,
    *,
    client: OpenAI,
    model: str,
    history: list[dict[str, str]] | None = None,
    max_rounds: int = 5,
) -> AgentResult:
    """Run a bounded model-tool-model conversation.

    ``history`` holds earlier user and assistant turns, oldest first; the
    caller keeps it, so the server stores nothing between requests.

    Tool failures never raise: they are returned to the model as structured
    errors. Errors from the model API itself still propagate to the caller.
    """
    normalized_text = user_text.strip()
    if not normalized_text:
        raise ValueError("user_text must not be empty")

    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_recent_history(history or []),
        {"role": "user", "content": normalized_text},
    ]

    tool_trace: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    consecutive_failures = 0
    rewrite_requested = False

    for round_index in range(max_rounds):
        # On the last round the model must answer from what it already has,
        # so collected tool results are not thrown away.
        is_last_round = round_index == max_rounds - 1
        if is_last_round and round_index > 0:
            # tool_choice="none" alone did not stop DeepSeek from printing a
            # tool call as text, so the model is also told in words.
            messages.append({"role": "user", "content": LAST_ROUND_INSTRUCTION})
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=AGENT_TOOLS,
            tool_choice="none" if is_last_round else "auto",
        )
        model_calls.append(
            _record_usage(response, model, round((time.perf_counter() - started) * 1000))
        )

        if not response.choices:
            raise RuntimeError("Model did not return a response")

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            answer = (message.content or "").strip()
            if not answer or has_leaked_tool_markup(answer):
                return AgentResult(DEGRADED_ANSWER, tool_trace, "degraded", model_calls)

            claims = find_unexecuted_action_claims(answer)
            field_names = find_internal_field_names(answer)
            if not claims and not field_names:
                return AgentResult(answer, tool_trace, "ok", model_calls)

            if rewrite_requested:
                # A false completion claim misleads the user and is blocked;
                # a leftover field name is only awkward wording, so the
                # answer is kept.
                if claims:
                    return AgentResult(BLOCKED_ANSWER, tool_trace, "blocked", model_calls)
                return AgentResult(answer, tool_trace, "ok", model_calls)

            rewrite_requested = True
            messages.append({"role": "assistant", "content": answer})
            messages.append(
                {
                    "role": "user",
                    "content": _rewrite_instruction(claims, field_names),
                }
            )
            continue

        messages.append(message)

        for tool_call in tool_calls:
            entry, tool_message = _run_tool_call(tool_call)
            tool_trace.append(entry)
            messages.append(tool_message)

            if entry["status"] == "error":
                consecutive_failures += 1
            else:
                consecutive_failures = 0

        if consecutive_failures >= MAX_CONSECUTIVE_TOOL_FAILURES:
            return AgentResult(DEGRADED_ANSWER, tool_trace, "degraded", model_calls)

    return AgentResult(DEGRADED_ANSWER, tool_trace, "degraded", model_calls)
