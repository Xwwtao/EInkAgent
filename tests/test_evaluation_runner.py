"""Tests for evaluation report generation."""

import json
from unittest.mock import Mock

from examples import evaluate_requirements as runner


def test_failed_request_is_recorded(tmp_path, monkeypatch):
    case_dir = tmp_path / "evals"
    case_dir.mkdir()
    (case_dir / "requirement_cases.json").write_text(
        json.dumps([
            {"id": "failed_case", "input": "预算2000元", "expected": {}}
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runner, "__file__", str(tmp_path / "examples" / "runner.py")
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")

    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(runner, "OpenAI", Mock(return_value=client))
    monkeypatch.setattr(
        runner, "parse_requirements", Mock(side_effect=RuntimeError("failed"))
    )

    runner.main()

    reports = list((case_dir / "runs").glob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))

    assert report["passed"] == 0
    assert report["total"] == 1
    assert len(report["results"]) == 1
    assert report["results"][0]["status"] == "ERROR"
    assert report["results"][0]["error_type"] == "RuntimeError"