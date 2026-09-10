"""Run live evaluations against labeled Agent cases."""

import json
import os
from pathlib import Path

from openai import OpenAI

from eink_agent.agent import run_agent
from eink_agent.agent_evaluation import (
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
    passed = 0

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

            if status == "PASS":
                passed += 1

            print(
                f'{case["id"]}: {status} '
                f'tools={tool_errors} '
                f'forbidden={forbidden_phrases}'
            )

    print(f"通过：{passed}/{len(cases)}")


if __name__ == "__main__":
    main()