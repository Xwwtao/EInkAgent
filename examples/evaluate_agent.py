"""Run live evaluations against labeled Agent cases."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from openai import OpenAI

from eink_agent.agent import SYSTEM_PROMPT, run_agent

from eink_agent.agent_evaluation import (
    build_agent_report,
    compare_tool_calls,
    find_forbidden_phrases,
)


def main() -> None:
    case_path = (
        Path(__file__).resolve().parents[1]
        / "evals"
        / "agent_cases.json"
    )
    cases = json.loads(case_path.read_text(encoding="utf-8"))
    model = os.environ["DEEPSEEK_MODEL"]
    started_at = datetime.now(timezone.utc)
    results = []

    with OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=0,
    ) as client:
        for case in cases:
            try:
                result = run_agent(
                    case["input"],
                    client=client,
                    model=model,
                )

                tool_errors = compare_tool_calls(
                    case["expected_calls"],
                    result.tool_trace,
                )
                forbidden_phrases = find_forbidden_phrases(
                    result.answer,
                    case["forbidden_answer_phrases"],
                )
            except Exception as error:
                results.append(
                    {
                        "id": case["id"],
                        "group": case["group"],
                        "input": case["input"],
                        "status": "ERROR",
                        "error": (
                            f"{type(error).__name__}: {error}"
                        ),
                    }
                )
                print(
                    f'{case["id"]}: ERROR '
                    f'({type(error).__name__})'
                )
                continue

            failed = (
                any(tool_errors.values())
                or bool(forbidden_phrases)
            )
            status = "FAIL" if failed else "PASS"

            actual_calls = [
                {
                    "name": trace["name"],
                    "arguments": trace["arguments"],
                    "result_count": trace["result_count"],
                }
                for trace in result.tool_trace
            ]

            results.append(
                {
                    "id": case["id"],
                    "group": case["group"],
                    "input": case["input"],
                    "expected_calls": case["expected_calls"],
                    "actual_calls": actual_calls,
                    "forbidden_answer_phrases": (
                        case["forbidden_answer_phrases"]
                    ),
                    "status": status,
                    "answer": result.answer,
                    "tool_errors": tool_errors,
                    "forbidden_phrase_matches": forbidden_phrases,
                }
            )

            print(
                f'{case["id"]}: {status} '
                f'tools={tool_errors} '
                f'forbidden={forbidden_phrases}'
            )

    report = build_agent_report(
        started_at=started_at.isoformat(),
        model=model,
        system_prompt=SYSTEM_PROMPT,
        results=results,
    )

    report_dir = case_path.parent / "agent_runs"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (
        started_at.strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"通过：{report['passed']}/{report['total']}")
    print(f"报告已保存：{report_path}")


if __name__ == "__main__":
    main()