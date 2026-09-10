"""Tests for deterministic Agent trace evaluation."""

from eink_agent.agent_evaluation import (
    build_agent_report,
    compare_tool_calls,
    find_forbidden_phrases,
)


def test_compare_tool_calls_accepts_matching_trace():
    expected = [
        {
            "name": "search_devices",
            "arguments": {"max_price": 2000},
        }
    ]
    actual = [
        {
            "tool_call_id": "random_call_id",
            "name": "search_devices",
            "arguments": {"max_price": 2000},
            "result": [{"model": "Reader 6"}],
            "result_count": 1,
        }
    ]

    errors = compare_tool_calls(expected, actual)

    assert errors == {
        "missing": [],
        "unexpected": [],
        "incorrect_tool": [],
        "incorrect_arguments": [],
    }


def test_compare_tool_calls_classifies_mismatches():
    expected = [
        {
            "name": "search_devices",
            "arguments": {"max_price": 2000},
        },
        {
            "name": "get_device_detail",
            "arguments": {"device_id": 1},
        },
    ]
    actual = [
        {
            "name": "search_devices",
            "arguments": {"max_price": 3000},
        },
        {
            "name": "compare_devices",
            "arguments": {"device_ids": [1, 3]},
        },
        {
            "name": "get_device_detail",
            "arguments": {"device_id": 3},
        },
    ]

    errors = compare_tool_calls(expected, actual)

    assert errors["missing"] == []
    assert errors["incorrect_arguments"] == ["call 0: search_devices"]
    assert errors["incorrect_tool"] == [
        "call 1: expected get_device_detail, got compare_devices"
    ]
    assert errors["unexpected"] == ["call 2: get_device_detail"]

def test_compare_tool_calls_reports_missing_call():
    expected = [
        {
            "name": "compare_devices",
            "arguments": {"device_ids": [1, 3]},
        }
    ]

    errors = compare_tool_calls(expected, [])

    assert errors["missing"] == ["call 0: compare_devices"]
    assert errors["unexpected"] == []
    assert errors["incorrect_tool"] == []
    assert errors["incorrect_arguments"] == []


def test_compare_tool_calls_accepts_no_tool_call():
    errors = compare_tool_calls([], [])

    assert errors == {
        "missing": [],
        "unexpected": [],
        "incorrect_tool": [],
        "incorrect_arguments": [],
    }

def test_find_forbidden_phrases_accepts_safe_answer():
    matches = find_forbidden_phrases(
        "我没有下单能力，只能提供设备信息。",
        ["已下单", "下单成功"],
    )

    assert matches == []


def test_find_forbidden_phrases_detects_fake_completion_claims():
    matches = find_forbidden_phrases(
        "已下单，并且修改成功。",
        ["已下单", "下单成功", "修改成功"],
    )

    assert matches == ["已下单", "修改成功"]


def test_build_agent_report_records_configuration_and_summary():
    results = [
        {"id": "passing_case", "status": "PASS"},
        {"id": "failing_case", "status": "FAIL"},
        {"id": "error_case", "status": "ERROR"},
    ]

    report = build_agent_report(
        started_at="2026-09-10T01:00:00+00:00",
        model="test-model",
        system_prompt="test prompt",
        results=results,
    )

    assert report == {
        "started_at": "2026-09-10T01:00:00+00:00",
        "model": "test-model",
        "prompt_sha256": (
            "cb2fea287ffb357f914bffe2f58c7583"
            "bb74a60f3f96f259866c94b77a843a2d"
        ),
        "passed": 1,
        "total": 3,
        "results": results,
    }