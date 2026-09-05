"""Run DeepSeek evaluations against labeled requirement cases."""

import json
import os
import hashlib

from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from eink_agent.requirement_evaluation import compare_requirements
from eink_agent.requirement_parser import SYSTEM_PROMPT, parse_requirements
from eink_agent.requirements import DeviceRequirements


def main() -> None:
    case_path = (
        Path(__file__).resolve().parents[1]
        / "evals"
        / "requirement_cases.json"
    )
    cases = json.loads(case_path.read_text(encoding="utf-8"))
    model = os.environ["DEEPSEEK_MODEL"]
    passed = 0
    started_at = datetime.now(timezone.utc)
    results = []

    with OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=0,
    ) as client:
        for case in cases:
            expected = DeviceRequirements.model_validate(case["expected"])
            record = {
                "id": case["id"],
                "input": case["input"],
                "expected": expected.model_dump(exclude_none=True),
            }

            try:
                actual = parse_requirements(
                    case["input"],
                    client=client,
                    model=model,
                )
                errors = compare_requirements(expected, actual)

            except Exception as error:
                record.update(
                    status="ERROR",
                    error_type=type(error).__name__,
                )
                results.append(record)
                print(f'{case["id"]}: ERROR ({type(error).__name__})')
                continue

            status = "FAIL" if any(errors.values()) else "PASS"
            record.update(
                status=status,
                actual=actual.model_dump(exclude_none=True),
                errors=errors,
            )
            results.append(record)

            if status == "PASS":
                passed += 1
            print(f'{case["id"]}: {status} {errors}')

    print(f"通过：{passed}/{len(cases)}")
    report = {
        "started_at": started_at.isoformat(),
        "model": model,
        "prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "passed": passed,
        "total": len(cases),
        "results": results,
    }

    report_dir = case_path.parent / "runs"
    report_dir.mkdir(exist_ok=True)
    filename = started_at.strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    report_path = report_dir / filename
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"报告已保存：{report_path}")


if __name__ == "__main__":
    main()