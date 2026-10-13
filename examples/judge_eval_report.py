"""Run the model judge over a saved evaluation report (decision 007).

    python -m examples.judge_eval_report evals/reports/A.json

Reads the questions, tool results and answers stored in the report; the
Agent is not run again. Writes judge_<report>.json and .md next to the
report, with calibration figures when evals/labels/<report>.json exists.
"""

import json
import os
from pathlib import Path
import sys
from typing import Any

from openai import OpenAI

from eink_agent.api import MODEL_MAX_RETRIES, MODEL_TIMEOUT_SECONDS
from eink_agent.llm_judge import calibrate, judge_answer

DEFAULT_JUDGE_MODEL = "qwen3.8-flash"
LABELS_DIR = Path(__file__).resolve().parents[1] / "evals" / "labels"


def _markdown(report: dict[str, Any], judged: dict[str, Any], calibration: dict[str, int] | None) -> str:
    verdicts = judged["verdicts"]
    ok = [v for v in verdicts.values() if "error" not in v]
    errors = len(verdicts) - len(ok)
    lines = [
        f"# 模型裁判：{judged['report']}",
        "",
        f"- 被评运行：{report['run']['started_at']}（Agent `{report['run']['model']}`）",
        f"- 裁判模型：`{judged['judge_model']}`，temperature 0，单次裁判",
        f"- 裁判条数：{len(verdicts)}（裁判出错 {errors} 条）",
        "",
        f"- grounded：{sum(v['grounded'] for v in ok)} / {len(ok)}",
        f"- responsive：{sum(v['responsive'] for v in ok)} / {len(ok)}",
        "",
    ]
    if calibration:
        lines += [
            "## 人工校准（grounded）",
            "",
            f"- 一致：{calibration['agreed']} / {calibration['compared']}",
            f"- 裁判漏判：{calibration['judge_missed']}（人工认定无依据，裁判判为有依据）",
            f"- 裁判误报：{calibration['judge_false_alarm']}（人工认定有依据，裁判判为无依据）",
            f"- 未比较：{calibration['skipped']}（裁判出错或未标注）",
            "",
        ]
    else:
        lines += ["尚无人工标注，未做校准；在校准之前不要引用上面的数字。", ""]
    lines += ["## 判为无依据或不回应的回答", "", "| 用例 | grounded | responsive | 无依据的陈述 |", "| --- | --- | --- | --- |"]
    for case_id, verdict in verdicts.items():
        if "error" in verdict:
            lines.append(f"| {case_id} | 错误 | 错误 | {verdict['error']} |")
        elif not verdict["grounded"] or not verdict["responsive"]:
            claims = "<br>".join(verdict["unsupported_claims"]).replace("|", "\\|") or "—"
            lines.append(f"| {case_id} | {verdict['grounded']} | {verdict['responsive']} | {claims} |")
    return "\n".join(lines) + "\n"


def run_judge(report_path: Path, *, client: Any, model: str, labels_dir: Path = LABELS_DIR) -> tuple[Path, Path]:
    """Judge every answered case in a report and write the judge files."""
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verdicts = {}
    for result in report["results"]:
        if result["answer"] is None:
            continue
        verdicts[result["id"]] = judge_answer(
            client=client,
            model=model,
            question=result["input"],
            history=result.get("history", []),
            tool_calls=result["tool_calls"],
            answer=result["answer"],
        )
        print(f"{result['id']}: {verdicts[result['id']]}")

    judged = {"report": report_path.name, "judge_model": model, "verdicts": verdicts}
    labels_path = Path(labels_dir) / report_path.name
    calibration = None
    if labels_path.exists():
        calibration = calibrate(verdicts, json.loads(labels_path.read_text(encoding="utf-8")))
        judged["calibration"] = calibration

    json_path = report_path.with_name(f"judge_{report_path.stem}.json")
    markdown_path = report_path.with_name(f"judge_{report_path.stem}.md")
    json_path.write_text(json.dumps(judged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report, judged, calibration), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    for name in ("JUDGE_API_KEY", "JUDGE_BASE_URL"):
        if not os.environ.get(name):
            raise SystemExit(f"未设置 {name}，见 .env.example。")
    with OpenAI(
        api_key=os.environ["JUDGE_API_KEY"],
        base_url=os.environ["JUDGE_BASE_URL"],
        timeout=MODEL_TIMEOUT_SECONDS,
        max_retries=MODEL_MAX_RETRIES,
    ) as client:
        json_path, markdown_path = run_judge(
            Path(sys.argv[1]), client=client, model=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        )
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"已保存：{json_path}\n      {markdown_path}")


if __name__ == "__main__":
    main()
