"""Grade a saved evaluation report again with the current cases.

    python -m examples.regrade_eval_report evals/reports/A.json

Uses the answers and tool calls stored in the report, so no model is
called and the original report file is left untouched. This isolates the
effect of a change to the grading rules from model randomness. Prints
Markdown and saves it as regrade_<report>.md next to the report.
"""

import json
from pathlib import Path
import sys

from eink_agent.agent_evaluation import grade_case

CASES_PATH = Path(__file__).resolve().parents[1] / "evals" / "agent_cases.json"


def regrade(report_path: Path, cases_path: Path = CASES_PATH) -> str:
    """Return Markdown listing how each saved result grades under current cases."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in json.loads(Path(cases_path).read_text(encoding="utf-8"))}

    changed = []
    passed_after = 0
    for result in report["results"]:
        run = {"answer": result["answer"], "status": result["status"], "tool_calls": result["tool_calls"]}
        if result["answer"] is None:
            run = {"exception": result["status"]}
        grade = grade_case(cases[result["id"]], run)
        passed_after += grade["result"] == "PASS"
        if grade["result"] != result["result"]:
            changed.append((result["id"], result["result"], grade["result"], grade["failures"]))

    total = len(report["results"])
    lines = [
        f"# 离线重判：{Path(report_path).name}",
        "",
        f"原报告运行于 {report['run']['started_at']}（`{report['run']['model']}`）。"
        "使用报告中保存的回答与工具调用，按当前用例文件重新判定，未调用模型，未修改原报告。",
        "",
        f"通过：{report['summary']['passed']} / {total} → {passed_after} / {total}",
        "",
        f"## 判定结果变化的用例（{len(changed)} 条）",
        "",
    ]
    if changed:
        lines += ["| 用例 | 变化 |", "| --- | --- |"]
        lines += [f"| {case_id} | {before} → {after} |" for case_id, before, after, _ in changed]
    else:
        lines.append("无。")
    return "\n".join(lines) + "\n"


def main() -> None:
    report_path = Path(sys.argv[1])
    markdown = regrade(report_path)
    output = report_path.with_name(f"regrade_{report_path.stem}.md")
    output.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"已保存：{output}")


if __name__ == "__main__":
    main()
