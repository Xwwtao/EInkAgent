"""Run a few live Agent requests and save what happened for review.

This is a manual smoke test, not an evaluation: there are no expected
answers. Results go to evals/smoke_runs/, which Git ignores.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from openai import OpenAI

from eink_agent.agent import run_agent

SMOKE_INPUTS = [
    "预算2000元以内，要能手写，推荐几款",
    "比较一下 Kindle Paperwhite 2026 和 Kobo Clara BW",
    "帮我下单 Kindle Colorsoft",
    "我想买个阅读器",
]


def _summarize_trace(tool_trace: list[dict]) -> list[dict]:
    """Keep the fields needed to review a run, without full results."""
    return [
        {
            "name": trace["name"],
            "arguments": trace["arguments"],
            "status": trace["status"],
            "error": trace.get("error"),
            "result_count": trace["result_count"],
        }
        for trace in tool_trace
    ]


def main() -> None:
    model = os.environ["DEEPSEEK_MODEL"]
    started_at = datetime.now(timezone.utc)
    runs = []

    with OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=0,
    ) as client:
        for user_text in SMOKE_INPUTS:
            start = time.perf_counter()
            try:
                result = run_agent(user_text, client=client, model=model)
            except Exception as error:
                runs.append(
                    {
                        "input": user_text,
                        "status": "exception",
                        "error": type(error).__name__,
                        "latency_ms": round((time.perf_counter() - start) * 1000),
                    }
                )
                print(f"[exception] {user_text} -> {type(error).__name__}")
                continue

            latency_ms = round((time.perf_counter() - start) * 1000)
            runs.append(
                {
                    "input": user_text,
                    "status": result.status,
                    "answer": result.answer,
                    "tool_calls": _summarize_trace(result.tool_trace),
                    "latency_ms": latency_ms,
                }
            )
            print(f"[{result.status}] {user_text} ({latency_ms} ms)")

    report_dir = Path(__file__).resolve().parents[1] / "evals" / "smoke_runs"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (started_at.strftime("%Y%m%dT%H%M%SZ") + ".json")
    report_path.write_text(
        json.dumps(
            {"started_at": started_at.isoformat(), "model": model, "runs": runs},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"结果已保存：{report_path}")


if __name__ == "__main__":
    main()
