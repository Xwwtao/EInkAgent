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
import tempfile
import time
from typing import Any
import uuid

import openai
from openai import OpenAI

from eink_agent.agent import MAX_HISTORY_MESSAGES, SYSTEM_PROMPT, AgentResult, run_agent
from eink_agent.agent_evaluation import grade_case, sum_usage, summarize
from eink_agent.api import (
    DEEPSEEK_BASE_URL,
    DEFAULT_MODEL,
    MODEL_MAX_RETRIES,
    MODEL_TIMEOUT_SECONDS,
    AgentRuntime,
    _knowledge_searcher,
    answer_in_session,
)
from eink_agent.memory import SQLiteMemoryStore, update_summary

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


def _result_ids(result: Any) -> list[int]:
    """Device IDs a tool returned, so cases can check which devices were found."""
    items = result if isinstance(result, list) else [result]
    return [item["id"] for item in items if isinstance(item, dict) and "id" in item]


def _result_docs(name: str, result: Any) -> list[str]:
    """Knowledge documents a search_knowledge call returned."""
    if name != "search_knowledge" or not isinstance(result, list):
        return []
    return list(dict.fromkeys(hit["chunk_id"].split("#")[0] for hit in result if isinstance(hit, dict)))


def _tool_call_views(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": call["name"],
            "arguments": call["arguments"],
            "status": call["status"],
            "error_type": call.get("error", {}).get("type") if call["status"] == "error" else None,
            "result_count": call["result_count"],
            "result_ids": _result_ids(call["result"]),
            "result_docs": _result_docs(call["name"], call["result"]),
            # Kept so the judge can grade later without rerunning the Agent.
            "result": call["result"],
        }
        for call in trace
    ]


def _run_conversation(
    case: dict[str, Any], client: Any, model: str, knowledge: Any, mode: str
) -> tuple[AgentResult, list[dict[str, Any]], list[dict[str, Any]]]:
    """Play the setup turns live, then the graded input (decision 009).

    ``memory``: each case gets a fresh temporary store and goes through
    answer_in_session, the code path of POST /agent with a session, plus the
    summary step that runs after each answer. ``window``: no memory, only
    the latest MAX_HISTORY_MESSAGES of the replayed conversation.
    Returns the final result, the setup turns and every model call.
    """
    setup: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []

    def record(text: str, result: AgentResult) -> None:
        setup.append({
            "input": text,
            "status": result.status,
            "answer": result.answer,
            "tool_calls": [
                {key: call[key] for key in ("name", "arguments", "status", "result_count")}
                for call in _tool_call_views(result.tool_trace)
            ],
        })
        model_calls.extend(result.model_calls)

    if mode == "window":
        history: list[dict[str, str]] = []
        for text in case["setup_turns"]:
            result = run_agent(text, client=client, model=model, history=history, knowledge=knowledge)
            record(text, result)
            if result.status == "ok":
                history += [{"role": "user", "content": text}, {"role": "assistant", "content": result.answer}]
        final = run_agent(case["input"], client=client, model=model, history=history, knowledge=knowledge)
        return final, setup, model_calls + final.model_calls

    runtime = AgentRuntime(client=client, model=model, knowledge=knowledge)
    session_id = str(uuid.uuid4())
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteMemoryStore(Path(directory) / "memory.db")
        for text in case["setup_turns"]:
            result, _ = answer_in_session(text, runtime=runtime, store=store, session_id=session_id)
            record(text, result)
            summary = update_summary(store, session_id, client=client, model=model, window=MAX_HISTORY_MESSAGES)
            if summary["model_call"]:
                model_calls.append(summary["model_call"])
        final, _ = answer_in_session(case["input"], runtime=runtime, store=store, session_id=session_id)
    return final, setup, model_calls + final.model_calls


def _run_case(
    case: dict[str, Any], client: Any, model: str, knowledge: Any = None, mode: str = "memory"
) -> dict[str, Any]:
    """Run one case; model errors are recorded instead of stopping the run.

    Cases with ``setup_turns`` are conversations, run in the given ``mode``.
    """
    started = time.perf_counter()
    setup: list[dict[str, Any]] = []
    try:
        if knowledge is not None and case.get("knowledge_extra_docs"):
            # Evaluation-only documents, e.g. one hiding an instruction.
            knowledge = knowledge.with_extra_documents(case["knowledge_extra_docs"])
        if "setup_turns" in case:
            result, setup, model_calls = _run_conversation(case, client, model, knowledge, mode)
        else:
            result = run_agent(
                case["input"], client=client, model=model, history=case["history"], knowledge=knowledge
            )
            model_calls = result.model_calls
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
            "rewrites": [],
            "setup_turns": setup,
        }

    tool_calls = _tool_call_views(result.tool_trace)
    return {
        "run": {"answer": result.answer, "status": result.status, "tool_calls": tool_calls},
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": sum_usage(model_calls),
        "answer": result.answer,
        "status": result.status,
        "tool_calls": tool_calls,
        "rewrites": result.rewrites,
        "setup_turns": setup,
    }


def _markdown(report: dict[str, Any]) -> str:
    run, summary = report["run"], report["summary"]
    cost = summary["cost_cny"]
    cost_text = (
        "未配置单价"
        if cost is None
        else " / ".join(f"{tier} ¥{value:.4f}" for tier, value in cost.items())
    )
    rewritten = "，".join(
        f"{result['id']}（{'、'.join(result['rewrites'])}）" for result in report["results"] if result["rewrites"]
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
        f"- 答案重写：{rewritten or '无'}",
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
    comparison = summary.get("memory_comparison")
    if comparison:
        window = {result["id"]: result for result in report["window_results"]}
        lines += [
            "",
            "## 记忆对比",
            "",
            "同一段对话跑两遍：有记忆（会话存储 + 偏好 + 摘要）与只有最近 "
            f"{MAX_HISTORY_MESSAGES} 条消息。只评最后一轮。",
            "",
            f"- 有记忆：{comparison['memory_passed']} / {comparison['cases']}；"
            f"只有窗口：{comparison['window_passed']} / {comparison['cases']}；"
            f"前几轮调用了 remember_preferences：{comparison['preference_writes']} / {comparison['cases']}",
            "",
            "| 用例 | 有记忆 | 只有窗口 | 写入偏好 |",
            "| --- | --- | --- | --- |",
            *[
                f"| {result['id']} | {result['result']} | {window[result['id']]['result']} | "
                f"{'是' if _wrote_preferences(result) else '否'} |"
                for result in report["results"]
                if result["id"] in window
            ],
        ]
    return "\n".join(lines) + "\n"


def _graded(case: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    grade = grade_case(case, outcome["run"])
    return {
        "id": case["id"],
        "category": case["category"],
        "input": case["input"],
        "history": case["history"],
        "setup_turns": outcome["setup_turns"],
        "result": grade["result"],
        "failures": grade["failures"],
        "status": outcome["status"],
        "answer": outcome["answer"],
        "tool_calls": outcome["tool_calls"],
        "rewrites": outcome["rewrites"],
        "latency_ms": outcome["latency_ms"],
        "usage": outcome["usage"],
    }


def _wrote_preferences(result: dict[str, Any]) -> bool:
    return any(
        call["name"] == "remember_preferences" and call["status"] == "ok"
        for turn in result["setup_turns"]
        for call in turn["tool_calls"]
    )


def _memory_comparison(results: list[dict[str, Any]], window_results: list[dict[str, Any]]) -> dict[str, int] | None:
    """Pass counts with memory and with the window only, for the same cases."""
    if not window_results:
        return None
    ids = {result["id"] for result in window_results}
    memory_results = [result for result in results if result["id"] in ids]
    return {
        "cases": len(window_results),
        "memory_passed": sum(result["result"] == "PASS" for result in memory_results),
        "window_passed": sum(result["result"] == "PASS" for result in window_results),
        "preference_writes": sum(_wrote_preferences(result) for result in memory_results),
    }


def run_evaluation(
    *,
    cases_path: Path,
    client: Any,
    model: str,
    report_dir: Path,
    pricing: dict[str, Any] | None,
    knowledge: Any = None,
) -> tuple[Path, Path]:
    """Run every case, grade it and write the JSON and Markdown reports."""
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    started_at = datetime.now(timezone.utc)
    results = []

    window_results = []

    for case in cases:
        results.append(_graded(case, _run_case(case, client, model, knowledge)))
        print(f"{case['id']}: {results[-1]['result']} " + " ".join(f["type"] for f in results[-1]["failures"]))
        if "setup_turns" in case:
            # The control: same conversation, only the recent-message window.
            window_results.append(_graded(case, _run_case(case, client, model, knowledge, mode="window")))
            print(f"{case['id']} (仅窗口): {window_results[-1]['result']}")

    summary = summarize(results, pricing)
    summary["memory_comparison"] = _memory_comparison(results, window_results)

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
            "knowledge": knowledge is not None,
        },
        "summary": summary,
        "results": results,
        "window_results": window_results,
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

    # The knowledge cases need search_knowledge; without it they would fail
    # for a configuration reason, not because of the Agent.
    knowledge = _knowledge_searcher()
    if knowledge is None:
        raise SystemExit("未设置 EMBEDDING_API_KEY / EMBEDDING_BASE_URL，知识问答用例需要它们，见 .env.example。")

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
                knowledge=knowledge,
            )
        except EvaluationAborted as error:
            raise SystemExit(f"评测已停止：{error}") from error

    report = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"通过：{report['summary']['passed']}/{report['summary']['total']}")
    print(f"报告：{json_path}\n      {markdown_path}")


if __name__ == "__main__":
    main()
