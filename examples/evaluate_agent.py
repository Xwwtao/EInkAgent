"""Run the Agent evaluation against DeepSeek and write committed reports.

    python -m examples.evaluate_agent

Writes evals/reports/<UTC time>.json and .md. Design:
docs/decisions/006-agent-evaluation.md.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import openai
from openai import OpenAI

from eink_agent.agent import SYSTEM_PROMPT, run_agent
from eink_agent.agent_evaluation import grade_case, sum_usage, summarize
from eink_agent.api import DEEPSEEK_BASE_URL, DEFAULT_MODEL, MODEL_MAX_RETRIES, MODEL_TIMEOUT_SECONDS

PROJECT_DIR = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_DIR / "evals" / "agent_cases.json"
PRICING_PATH = PROJECT_DIR / "evals" / "pricing.json"
REPORT_DIR = PROJECT_DIR / "evals" / "reports"
DATA_FILES = (
    PROJECT_DIR / "data" / "real_devices.csv",
    PROJECT_DIR / "data" / "real_offers.csv",
)


class EvaluationAborted(RuntimeError):
    """A configuration error that would fail every case, so the run stops."""


# These mean the key or model name is wrong: retrying or running the other
# cases cannot succeed, and a report full of them would measure nothing.
CONFIGURATION_ERRORS = {
    openai.AuthenticationError: "模型服务拒绝了认证（HTTP 401），请检查 DEEPSEEK_API_KEY",
    openai.PermissionDeniedError: "模型服务拒绝访问（HTTP 403），请检查 DEEPSEEK_API_KEY 的权限",
    openai.NotFoundError: "模型服务找不到该模型（HTTP 404），请检查 DEEPSEEK_MODEL",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    """Return the current commit, marked dirty when files are modified."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_DIR, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_DIR, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def _run_case(case: dict[str, Any], client: Any, model: str) -> dict[str, Any]:
    """Run one case; model errors are recorded instead of stopping the run."""
    started = time.perf_counter()
    try:
        result = run_agent(case["input"], client=client, model=model, history=case["history"])
    except tuple(CONFIGURATION_ERRORS) as error:
        message = next(text for kind, text in CONFIGURATION_ERRORS.items() if isinstance(error, kind))
        raise EvaluationAborted(f"{message}；本次不生成报告。") from error
    except Exception as error:  # noqa: BLE001 - every failure must stay in the report
        return {
            "run": {"exception": type(error).__name__},
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "usage": sum_usage([]),
            "answer": None,
            "status": "exception",
            "tool_calls": [],
        }

    tool_calls = [
        {
            "name": call["name"],
            "arguments": call["arguments"],
            "status": call["status"],
            "error_type": call.get("error", {}).get("type") if call["status"] == "error" else None,
            "result_count": call["result_count"],
        }
        for call in result.tool_trace
    ]
    return {
        "run": {"answer": result.answer, "status": result.status, "tool_calls": tool_calls},
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": sum_usage(result.model_calls),
        "answer": result.answer,
        "status": result.status,
        "tool_calls": tool_calls,
    }


def _markdown(report: dict[str, Any]) -> str:
    run, summary = report["run"], report["summary"]
    cost = summary["cost_cny"]
    cost_text = (
        "未配置单价"
        if cost is None
        else " / ".join(f"{tier} ¥{value:.4f}" for tier, value in cost.items())
    )
    lines = [
        f"# Agent 评测报告 {run['started_at']}",
        "",
        f"- 模型：`{run['model']}`；git 提交：`{run['git_commit']}`；单次运行",
        f"- 样本量：{run['sample_size']} 条（`{run['cases_file']}`）",
        f"- 数据快照 SHA-256：" + "，".join(f"`{name}` {digest[:12]}" for name, digest in run["data_sha256"].items()),
        f"- 系统提示词 SHA-256：{run['prompt_sha256'][:12]}",
        "",
        "## 结果",
        "",
        f"- 通过：**{summary['passed']} / {summary['total']}**",
        f"- 延迟（每条用例端到端）：p50 {summary['latency_ms']['p50']} ms，p95 {summary['latency_ms']['p95']} ms",
        f"- token：输入 {summary['tokens']['prompt']}，输出 {summary['tokens']['completion']}",
        f"- 成本估算：{cost_text}（单价见 `evals/pricing.json`）",
        "",
        "| 类别 | 通过 |",
        "| --- | --- |",
        *[f"| {name} | {value['passed']} / {value['total']} |" for name, value in summary["by_category"].items()],
        "",
        "| 失败类型 | 次数 |",
        "| --- | --- |",
        *([f"| {name} | {count} |" for name, count in summary["failure_types"].items()] or ["| 无 | 0 |"]),
        "",
        "## 逐条结果",
        "",
        "| 用例 | 类别 | 结果 | 工具调用 | 耗时 ms | 失败原因 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in report["results"]:
        calls = "<br>".join(
            f"{call['name']}→{call['result_count'] if call['status'] == 'ok' else call['error_type']}"
            for call in result["tool_calls"]
        ) or "—"
        reasons = "<br>".join(f"{f['type']}: {f['detail']}" for f in result["failures"]).replace("|", "\\|") or "—"
        lines.append(
            f"| {result['id']} | {result['category']} | {result['result']} | {calls} | {result['latency_ms']} | {reasons} |"
        )
    return "\n".join(lines) + "\n"


def run_evaluation(
    *,
    cases_path: Path,
    client: Any,
    model: str,
    report_dir: Path,
    pricing: dict[str, Any] | None,
) -> tuple[Path, Path]:
    """Run every case, grade it and write the JSON and Markdown reports."""
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    started_at = datetime.now(timezone.utc)
    results = []

    for case in cases:
        outcome = _run_case(case, client, model)
        grade = grade_case(case, outcome["run"])
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "input": case["input"],
                "result": grade["result"],
                "failures": grade["failures"],
                "status": outcome["status"],
                "answer": outcome["answer"],
                "tool_calls": outcome["tool_calls"],
                "latency_ms": outcome["latency_ms"],
                "usage": outcome["usage"],
            }
        )
        print(f"{case['id']}: {grade['result']} " + " ".join(f["type"] for f in grade["failures"]))

    report = {
        "run": {
            "started_at": started_at.isoformat(),
            "model": model,
            "git_commit": _git_commit(),
            "sample_size": len(cases),
            "cases_file": cases_path.name,
            "cases_sha256": _sha256(cases_path),
            "data_sha256": {path.name: _sha256(path) for path in DATA_FILES},
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "pricing": pricing,
        },
        "summary": summarize(results, pricing),
        "results": results,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    stem = started_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"{stem}.json"
    markdown_path = report_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
    pricing = json.loads(PRICING_PATH.read_text(encoding="utf-8")) if PRICING_PATH.exists() else None
    if pricing and pricing.get("model") != model:
        print(f"注意：evals/pricing.json 是 {pricing.get('model')} 的单价，本次模型为 {model}，不估算成本。")
        pricing = None

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("未设置 DEEPSEEK_API_KEY，请先在当前终端设置后再运行。")

    with OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=DEEPSEEK_BASE_URL,
        timeout=MODEL_TIMEOUT_SECONDS,
        max_retries=MODEL_MAX_RETRIES,
    ) as client:
        try:
            json_path, markdown_path = run_evaluation(
                cases_path=CASES_PATH,
                client=client,
                model=model,
                report_dir=REPORT_DIR,
                pricing=pricing,
            )
        except EvaluationAborted as error:
            raise SystemExit(f"评测已停止：{error}") from error

    report = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"通过：{report['summary']['passed']}/{report['summary']['total']}")
    print(f"报告：{json_path}\n      {markdown_path}")


if __name__ == "__main__":
    main()
