"""Tests for grading Agent runs against evaluation cases.

These replace the exact-sequence comparison used before decision 006:
grading now checks required calls by argument subset and result count.
"""

import json
from pathlib import Path

import pytest

from eink_agent.agent_evaluation import (
    CATEGORIES,
    estimate_cost,
    grade_case,
    percentile,
    summarize,
)

CASES_PATH = Path(__file__).resolve().parents[1] / "evals" / "agent_cases.json"
PRICING = {
    "tiers": {
        "off_peak": {"input_cache_hit": 0.02, "input_cache_miss": 1, "output": 4},
        "peak": {"input_cache_hit": 0.04, "input_cache_miss": 2, "output": 8},
    }
}


def _case(**expect) -> dict:
    base = {
        "status": "ok",
        "required_calls_any_of": [],
        "max_tool_calls": 6,
        "no_tool_calls": False,
        "answer_includes_any": [],
        "answer_includes_all": [],
        "answer_excludes": [],
    }
    base.update(expect)
    return {"id": "case", "category": "single_search", "input": "x", "history": [], "expect": base}


def _call(name: str, arguments: dict, result_count: int, status: str = "ok", ids=None) -> dict:
    return {
        "name": name,
        "arguments": arguments,
        "status": status,
        "result_count": result_count,
        "result_ids": ids or [],
    }


def _run(answer="好的。", tool_calls=None, status="ok") -> dict:
    return {"answer": answer, "status": status, "tool_calls": tool_calls or []}


BUDGET_STYLUS = _case(
    required_calls_any_of=[
        [{"name": "search_devices", "arguments": {"max_price": 2000, "supports_stylus": True}, "result_count": 2}]
    ]
)


def _failure_types(grade: dict) -> list[str]:
    return [failure["type"] for failure in grade["failures"]]


def test_required_call_with_extra_arguments_and_calls_passes():
    run = _run(
        tool_calls=[
            _call("search_devices", {"max_price": 2000, "supports_stylus": True, "limit": 20}, 2),
            _call("get_device_detail", {"device_id": 17}, 1),
        ]
    )

    assert grade_case(BUDGET_STYLUS, run) == {"result": "PASS", "failures": []}


def test_sabotaged_query_returning_zero_rows_fails():
    """A wrong budget changes the row count, so the case must fail."""
    run = _run(tool_calls=[_call("search_devices", {"max_price": 2000, "supports_stylus": True}, 0)])

    grade = grade_case(BUDGET_STYLUS, run)

    assert grade["result"] == "FAIL"
    assert _failure_types(grade) == ["wrong_result_count"]


def test_wrong_argument_value_is_a_missing_call():
    run = _run(tool_calls=[_call("search_devices", {"max_price": 200, "supports_stylus": True}, 0)])

    grade = grade_case(BUDGET_STYLUS, run)

    assert _failure_types(grade) == ["missing_call"]


def test_failed_tool_call_does_not_satisfy_a_requirement():
    run = _run(
        tool_calls=[
            _call("search_devices", {"max_price": 2000, "supports_stylus": True}, 0, status="error")
        ]
    )

    assert _failure_types(grade_case(BUDGET_STYLUS, run)) == ["missing_call"]


def test_list_arguments_are_compared_as_sets():
    case = _case(
        required_calls_any_of=[
            [{"name": "compare_devices", "arguments": {"device_ids": [45, 15]}, "result_count": 2}]
        ]
    )
    run = _run(tool_calls=[_call("compare_devices", {"device_ids": [15, 45]}, 2)])

    assert grade_case(case, run)["result"] == "PASS"


def test_any_alternative_call_set_is_enough():
    case = _case(
        required_calls_any_of=[
            [{"name": "compare_devices", "arguments": {"device_ids": [45, 15]}, "result_count": 2}],
            [
                {"name": "get_device_detail", "arguments": {"device_id": 45}, "result_count": 1},
                {"name": "get_device_detail", "arguments": {"device_id": 15}, "result_count": 1},
            ],
        ]
    )
    run = _run(
        tool_calls=[
            _call("get_device_detail", {"device_id": 15}, 1),
            _call("get_device_detail", {"device_id": 45}, 1),
        ]
    )

    assert grade_case(case, run)["result"] == "PASS"


def test_clarification_case_fails_when_a_tool_is_called():
    case = _case(no_tool_calls=True)
    run = _run(tool_calls=[_call("search_devices", {}, 20)])

    assert _failure_types(grade_case(case, run)) == ["unexpected_tool_call"]


def test_too_many_calls_fail():
    case = _case(max_tool_calls=2)
    run = _run(tool_calls=[_call("search_devices", {}, 1)] * 3)

    assert _failure_types(grade_case(case, run)) == ["too_many_calls"]


def test_status_mismatch_fails():
    run = _run(
        status="degraded",
        tool_calls=[_call("search_devices", {"max_price": 2000, "supports_stylus": True}, 2)],
    )

    assert _failure_types(grade_case(BUDGET_STYLUS, run)) == ["status_mismatch"]


@pytest.mark.parametrize(
    ("expect", "answer", "failure"),
    [
        ({"answer_includes_any": ["没有", "未找到"]}, "找到 3 款。", "answer_missing"),
        ({"answer_includes_all": ["192", "200"]}, "Leaf5+ 是 192 克。", "answer_missing"),
        ({"answer_excludes": ["1元"]}, "现价 1元。", "answer_forbidden"),
        ({}, "已为您下单。", "action_claim"),
    ],
)
def test_answer_checks(expect, answer, failure):
    assert _failure_types(grade_case(_case(**expect), _run(answer=answer))) == [failure]


def test_exception_run_fails_with_exception_type():
    grade = grade_case(BUDGET_STYLUS, {"exception": "APITimeoutError"})

    assert grade == {
        "result": "FAIL",
        "failures": [{"type": "exception", "detail": "APITimeoutError"}],
    }


def test_percentile_uses_nearest_rank():
    values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    assert percentile(values, 50) == 500
    assert percentile(values, 95) == 1000
    assert percentile([], 50) is None


def test_estimate_cost_gives_off_peak_and_peak_bounds():
    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
        "prompt_cache_hit_tokens": None,
        "prompt_cache_miss_tokens": None,
    }

    # Without cache figures every prompt token is priced as a cache miss.
    assert estimate_cost(usage, PRICING) == {"off_peak": 5.0, "peak": 10.0}


def test_estimate_cost_uses_cache_split_when_reported():
    usage = {
        "prompt_tokens": 2_000_000,
        "completion_tokens": 0,
        "prompt_cache_hit_tokens": 1_000_000,
        "prompt_cache_miss_tokens": 1_000_000,
    }

    assert estimate_cost(usage, PRICING) == {"off_peak": 1.02, "peak": 2.04}


def test_estimate_cost_without_pricing_is_none():
    assert estimate_cost({"prompt_tokens": 1, "completion_tokens": 1}, None) is None


def test_summarize_counts_by_category_and_failure_type():
    results = [
        {"category": "single_search", "result": "PASS", "failures": [], "latency_ms": 100,
         "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        {"category": "single_search", "result": "FAIL", "failures": [{"type": "missing_call"}],
         "latency_ms": 300, "usage": {"prompt_tokens": 20, "completion_tokens": 5}},
        {"category": "no_result", "result": "FAIL", "failures": [{"type": "exception"}],
         "latency_ms": 200, "usage": {"prompt_tokens": 0, "completion_tokens": 0}},
    ]

    summary = summarize(results, pricing=None)

    assert summary["total"] == 3
    assert summary["passed"] == 1
    assert summary["by_category"]["single_search"] == {"passed": 1, "total": 2}
    assert summary["failure_types"] == {"exception": 1, "missing_call": 1}
    assert summary["latency_ms"] == {"p50": 200, "p95": 300}
    assert summary["tokens"] == {"prompt": 30, "completion": 10}
    assert summary["cost_cny"] is None


def test_case_file_is_well_formed():
    """Catch typos in evals/agent_cases.json before a paid run."""
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    tool_names = {"search_devices", "get_device_detail", "compare_devices"}

    assert len(cases) >= 30
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["category"] in CATEGORIES, case["id"]
        assert case["input"].strip(), case["id"]
        for option in case["expect"]["required_calls_any_of"]:
            for requirement in option:
                assert set(requirement) <= {"name", "arguments", "result_count", "result_ids_include"}, case["id"]
                assert requirement.get("name", "search_devices") in tool_names, case["id"]
                assert "result_count" in requirement or requirement.get("result_ids_include"), case["id"]
        if case["expect"]["no_tool_calls"]:
            assert not case["expect"]["required_calls_any_of"], case["id"]


def test_found_devices_pass_regardless_of_keyword_wording():
    case = _case(required_calls_any_of=[[{"result_ids_include": [45]}, {"result_ids_include": [46]}]])
    run = _run(tool_calls=[_call("search_devices", {"keyword": "Paperwhite"}, 4, ids=[45, 46, 23, 24])])

    assert grade_case(case, run)["result"] == "PASS"


def test_devices_found_across_separate_calls_pass():
    case = _case(required_calls_any_of=[[{"result_ids_include": [38]}, {"result_ids_include": [17]}]])
    run = _run(
        tool_calls=[
            _call("search_devices", {"keyword": "Leaf5+"}, 1, ids=[38]),
            _call("get_device_detail", {"device_id": 17}, 1, ids=[17]),
        ]
    )

    assert grade_case(case, run)["result"] == "PASS"


def test_missing_device_is_reported_as_wrong_devices():
    case = _case(required_calls_any_of=[[{"name": "search_devices", "result_ids_include": [17]}]])
    run = _run(tool_calls=[_call("search_devices", {"keyword": "Libra"}, 1, ids=[99])])

    grade = grade_case(case, run)

    assert _failure_types(grade) == ["wrong_devices"]
    assert "17" in grade["failures"][0]["detail"]


def test_named_requirement_ignores_other_tools():
    case = _case(required_calls_any_of=[[{"name": "compare_devices", "result_ids_include": [16, 20]}]])
    run = _run(tool_calls=[_call("search_devices", {"is_color": True}, 7, ids=[16, 20, 17])])

    assert _failure_types(grade_case(case, run)) == ["missing_call"]
