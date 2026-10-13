"""Compare two evaluation reports: which cases changed and how metrics moved.

    python -m examples.compare_eval_reports evals/reports/A.json evals/reports/B.json

Prints Markdown and saves it next to the second report.
"""

import json
from pathlib import Path
import sys


def compare(first_path: Path, second_path: Path) -> str:
    """Return a Markdown comparison of two report files."""
    first = json.loads(Path(first_path).read_text(encoding="utf-8"))
    second = json.loads(Path(second_path).read_text(encoding="utf-8"))

    before = {result["id"]: result["result"] for result in first["results"]}
    after = {result["id"]: result["result"] for result in second["results"]}
    changed = [
        (case_id, before.get(case_id, "—"), after.get(case_id, "—"))
        for case_id in sorted(before.keys() | after.keys())
        if before.get(case_id) != after.get(case_id)
    ]

    def metrics(report):
        summary = report["summary"]
        return (
            f"{summary['passed']} / {summary['total']}",
            summary["latency_ms"]["p50"],
            summary["latency_ms"]["p95"],
            summary["tokens"]["prompt"] + summary["tokens"]["completion"],
        )

    rows = zip(("通过", "p50 延迟 ms", "p95 延迟 ms", "token 合计"), metrics(first), metrics(second))
    lines = [
        f"# 评测对比：{first['run']['started_at']} → {second['run']['started_at']}",
        "",
        f"模型：`{first['run']['model']}` → `{second['run']['model']}`",
        "",
        "| 指标 | 前一次 | 后一次 |",
        "| --- | --- | --- |",
        *[f"| {name} | {a} | {b} |" for name, a, b in rows],
        "",
        f"## 结果变化的用例（{len(changed)} 条）",
        "",
    ]
    if changed:
        lines += ["| 用例 | 变化 |", "| --- | --- |"]
        lines += [f"| {case_id} | {a} → {b} |" for case_id, a, b in changed]
    else:
        lines.append("无。")
    return "\n".join(lines) + "\n"


def main() -> None:
    first, second = (Path(arg) for arg in sys.argv[1:3])
    markdown = compare(first, second)
    output = second.with_name(f"compare_{first.stem}_vs_{second.stem}.md")
    output.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"已保存：{output}")


if __name__ == "__main__":
    main()
