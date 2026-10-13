"""Grade Agent runs against evaluation cases and summarize the results.

Every function here is deterministic and never calls a model, so grading can
be unit tested. Design: docs/decisions/006-agent-evaluation.md.
"""

from collections import Counter
import math
from typing import Any

from eink_agent.answer_guard import find_unexecuted_action_claims

CATEGORIES = (
    "single_search",
    "multi_step",
    "no_result",
    "invalid_params",
    "clarification",
    "unsupported_action",
    "prompt_injection",
    "multi_turn",
    "data_honesty",
)

TOKENS_PER_PRICE_UNIT = 1_000_000


def _same_value(expected: Any, actual: Any) -> bool:
    """Compare argument values; lists match regardless of order."""
    if isinstance(expected, list) and isinstance(actual, list):
        return sorted(expected) == sorted(actual)
    return expected == actual


def _arguments_match(expected: dict[str, Any], actual: dict[str, Any] | None) -> bool:
    """Every expected key must be present and equal; extra keys are allowed."""
    if actual is None:
        return False
    return all(
        key in actual and _same_value(value, actual[key])
        for key, value in expected.items()
    )


def _describe(requirement: dict[str, Any]) -> str:
    parts = [requirement.get("name", "any tool")]
    if requirement.get("arguments"):
        parts.append(str(requirement["arguments"]))
    if requirement.get("result_ids_include"):
        parts.append(f"finding devices {requirement['result_ids_include']}")
    return " ".join(parts)


def _check_option(option: list[dict[str, Any]], calls: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return the failures for one set of required calls (empty = satisfied).

    A requirement may name a tool and arguments, and checks the outcome by
    row count, by which devices were returned, or both. Without a name, any
    tool counts: "find device 45" can be met by a search or a detail call.
    """
    failures = []

    for requirement in option:
        label = _describe(requirement)
        candidates = [
            call
            for call in calls
            if call["status"] == "ok"
            and call["name"] == requirement.get("name", call["name"])
            and _arguments_match(requirement.get("arguments", {}), call["arguments"])
        ]
        if not candidates:
            failures.append({"type": "missing_call", "detail": label})
            continue

        wanted_ids = set(requirement.get("result_ids_include", []))
        if wanted_ids:
            with_ids = [call for call in candidates if wanted_ids <= set(call.get("result_ids", []))]
            if not with_ids:
                found = sorted({i for call in candidates for i in call.get("result_ids", [])})
                failures.append(
                    {"type": "wrong_devices", "detail": f"{label}: found {found[:10]}"}
                )
                continue
            candidates = with_ids

        if "result_count" in requirement and all(
            call["result_count"] != requirement["result_count"] for call in candidates
        ):
            counts = sorted({call["result_count"] for call in candidates})
            failures.append(
                {
                    "type": "wrong_result_count",
                    "detail": f"{label}: expected {requirement['result_count']}, got {counts}",
                }
            )

    return failures


def grade_case(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Grade one run. ``run`` has answer, status and tool_calls, or exception."""
    if "exception" in run:
        return {
            "result": "FAIL",
            "failures": [{"type": "exception", "detail": run["exception"]}],
        }

    expect = case["expect"]
    calls = run["tool_calls"]
    answer = run["answer"]
    failures: list[dict[str, str]] = []

    if run["status"] != expect["status"]:
        failures.append(
            {"type": "status_mismatch", "detail": f"expected {expect['status']}, got {run['status']}"}
        )

    if expect["no_tool_calls"] and calls:
        failures.append(
            {"type": "unexpected_tool_call", "detail": ", ".join(call["name"] for call in calls)}
        )

    if len(calls) > expect["max_tool_calls"]:
        failures.append(
            {"type": "too_many_calls", "detail": f"{len(calls)} > {expect['max_tool_calls']}"}
        )

    options = expect["required_calls_any_of"]
    if options:
        option_failures = [_check_option(option, calls) for option in options]
        if all(option_failures):
            # Report the closest alternative: the one with the fewest failures.
            failures.extend(min(option_failures, key=len))

    if expect["answer_includes_any"] and not any(
        phrase in answer for phrase in expect["answer_includes_any"]
    ):
        failures.append(
            {"type": "answer_missing", "detail": "none of " + " / ".join(expect["answer_includes_any"])}
        )

    missing = [phrase for phrase in expect["answer_includes_all"] if phrase not in answer]
    if missing:
        failures.append({"type": "answer_missing", "detail": "missing " + " / ".join(missing)})

    forbidden = [phrase for phrase in expect["answer_excludes"] if phrase in answer]
    if forbidden:
        failures.append({"type": "answer_forbidden", "detail": " / ".join(forbidden)})

    claims = find_unexecuted_action_claims(answer)
    if claims:
        failures.append({"type": "action_claim", "detail": " / ".join(claims)})

    return {"result": "FAIL" if failures else "PASS", "failures": failures}


def percentile(values: list[int], rank: int) -> int | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(math.ceil(rank / 100 * len(ordered)), 1) - 1
    return ordered[index]


def estimate_cost(usage: dict[str, Any], pricing: dict[str, Any] | None) -> dict[str, float] | None:
    """Price token usage at every tier; None when no pricing is configured.

    Without a cache split from the provider, every prompt token is priced as
    a cache miss, which gives an upper bound.
    """
    if not pricing:
        return None

    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is None or miss is None:
        hit, miss = 0, prompt

    return {
        tier: round(
            (hit * prices["input_cache_hit"] + miss * prices["input_cache_miss"]
             + completion * prices["output"]) / TOKENS_PER_PRICE_UNIT,
            6,
        )
        for tier, prices in pricing["tiers"].items()
    }


def sum_usage(model_calls: list[dict[str, Any]]) -> dict[str, int | None]:
    """Add up token usage across model calls; cache figures only if all report them."""
    def total(name: str) -> int | None:
        values = [call.get(name) for call in model_calls]
        if not values or any(value is None for value in values):
            return None
        return sum(values)

    return {
        "prompt_tokens": sum(call.get("prompt_tokens") or 0 for call in model_calls),
        "completion_tokens": sum(call.get("completion_tokens") or 0 for call in model_calls),
        "prompt_cache_hit_tokens": total("prompt_cache_hit_tokens"),
        "prompt_cache_miss_tokens": total("prompt_cache_miss_tokens"),
    }


def summarize(results: list[dict[str, Any]], pricing: dict[str, Any] | None) -> dict[str, Any]:
    """Aggregate pass rates, failure types, latency, tokens and cost."""
    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_category.setdefault(result["category"], {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += result["result"] == "PASS"

    failure_types = Counter(
        failure["type"] for result in results for failure in result["failures"]
    )
    latencies = [result["latency_ms"] for result in results]
    usage = {
        "prompt_tokens": sum(result["usage"].get("prompt_tokens") or 0 for result in results),
        "completion_tokens": sum(result["usage"].get("completion_tokens") or 0 for result in results),
    }
    hits = [result["usage"].get("prompt_cache_hit_tokens") for result in results]
    misses = [result["usage"].get("prompt_cache_miss_tokens") for result in results]
    usage["prompt_cache_hit_tokens"] = None if None in hits else sum(hits)
    usage["prompt_cache_miss_tokens"] = None if None in misses else sum(misses)

    return {
        "total": len(results),
        "passed": sum(result["result"] == "PASS" for result in results),
        "by_category": by_category,
        "failure_types": dict(sorted(failure_types.items())),
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "tokens": {"prompt": usage["prompt_tokens"], "completion": usage["completion_tokens"]},
        "cost_cny": estimate_cost(usage, pricing),
    }
