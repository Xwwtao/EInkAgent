"""End-to-end tests for the Agent evaluation runner, with a fake model."""

import json
from unittest.mock import Mock

import httpx
import openai

from examples import compare_eval_reports, evaluate_agent

CASES = [
    {
        "id": "s01",
        "category": "single_search",
        "input": "Reader 6 多少钱",
        "history": [],
        "expect": {
            "status": "ok",
            "required_calls_any_of": [
                [{"name": "search_devices", "arguments": {"keyword": "Reader"}, "result_count": 1}]
            ],
            "max_tool_calls": 6,
            "no_tool_calls": False,
            "answer_includes_any": ["899"],
            "answer_includes_all": [],
            "answer_excludes": [],
        },
    },
    {
        "id": "c01",
        "category": "clarification",
        "input": "推荐一款",
        "history": [],
        "expect": {
            "status": "ok",
            "required_calls_any_of": [],
            "max_tool_calls": 6,
            "no_tool_calls": True,
            "answer_includes_any": [],
            "answer_includes_all": [],
            "answer_excludes": [],
        },
    },
]


def _response(content=None, tool_calls=None, prompt_tokens=100, completion_tokens=10):
    response = Mock(choices=[Mock(message=Mock(content=content, tool_calls=tool_calls or []))])
    response.usage = Mock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        spec=["prompt_tokens", "completion_tokens"],
    )
    return response


def _tool_call(name, arguments, call_id):
    function = Mock(arguments=json.dumps(arguments))
    function.name = name
    return Mock(id=call_id, function=function)


def _run(tmp_path, client):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(CASES, ensure_ascii=False), encoding="utf-8")
    return evaluate_agent.run_evaluation(
        cases_path=cases_path,
        client=client,
        model="fake-model",
        report_dir=tmp_path / "reports",
        pricing=None,
    )


def test_runner_grades_cases_and_writes_json_and_markdown(tmp_path):
    client = Mock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call("search_devices", {"keyword": "Reader"}, "c1")]),
        _response(content="Reader 6 参考价 899 元。"),
        _response(content="请问预算和用途是什么？"),
    ]

    json_path, markdown_path = _run(tmp_path, client)

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["run"]["model"] == "fake-model"
    assert report["run"]["sample_size"] == 2
    assert len(report["run"]["data_sha256"]) == 2
    assert report["summary"]["passed"] == 2
    assert report["summary"]["tokens"] == {"prompt": 300, "completion": 30}
    assert report["results"][0]["tool_calls"][0]["result_count"] == 1
    assert "result" not in report["results"][0]["tool_calls"][0]

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "fake-model" in markdown
    assert "2 / 2" in markdown


def test_model_error_is_counted_as_failed_case(tmp_path):
    client = Mock()
    client.chat.completions.create.side_effect = openai.APITimeoutError(
        request=httpx.Request("POST", "https://api.deepseek.com")
    )

    json_path, _ = _run(tmp_path, client)

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["summary"]["passed"] == 0
    assert report["summary"]["total"] == 2
    assert report["summary"]["failure_types"] == {"exception": 2}


def test_compare_reports_lists_changed_cases(tmp_path):
    def report(results, passed):
        return {
            "run": {"started_at": "t", "model": "m"},
            "summary": {"passed": passed, "total": 2, "latency_ms": {"p50": 1, "p95": 2},
                        "tokens": {"prompt": 1, "completion": 1}},
            "results": results,
        }

    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(report([{"id": "x", "result": "PASS"}, {"id": "y", "result": "FAIL"}], 1)))
    second.write_text(json.dumps(report([{"id": "x", "result": "FAIL"}, {"id": "y", "result": "FAIL"}], 0)))

    markdown = compare_eval_reports.compare(first, second)

    assert "x" in markdown and "PASS → FAIL" in markdown
    assert "| y |" not in markdown
