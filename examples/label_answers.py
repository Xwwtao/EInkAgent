"""Hand-label whether each answer in a report is grounded (decision 007).

    python -m examples.label_answers evals/reports/A.json

Shows each question, the tool results and the answer, then asks y/n:
"is every device fact in the answer supported by the tool results?"
Labels are saved after every answer to evals/labels/<report>.json, so the
session can be stopped and resumed. The judge's verdicts are not shown.
"""

import json
from pathlib import Path
import sys
from typing import Any, Callable

LABELS_DIR = Path(__file__).resolve().parents[1] / "evals" / "labels"
QUESTION = "回答里关于设备的每条事实，都能在上面的工具结果中找到依据吗？(y/n，q 退出) "


def _format(result: dict[str, Any]) -> str:
    lines = [f"\n===== {result['id']}（{result['category']}）=====", f"问题：{result['input']}"]
    for turn in result.get("history", []):
        lines.append(f"  [历史 {turn['role']}] {turn['content']}")
    for call in result["tool_calls"]:
        lines.append(f"工具 {call['name']} {json.dumps(call['arguments'], ensure_ascii=False)}:")
        if call["status"] != "ok":
            lines.append(f"  （失败：{call.get('error_type')}）")
            continue
        items = call["result"] if isinstance(call["result"], list) else [call["result"]]
        for item in items:
            if isinstance(item, dict):
                facts = ", ".join(f"{k}={v}" for k, v in item.items() if k not in ("spec_source_url", "price_url", "notes"))
                lines.append(f"  - {facts}")
    lines.append(f"回答：\n{result['answer']}")
    return "\n".join(lines)


def label(
    report_path: Path,
    *,
    labels_dir: Path = LABELS_DIR,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> Path:
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels_path = labels_dir / report_path.name
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}

    for result in report["results"]:
        if result["answer"] is None or result["id"] in labels:
            continue
        show(_format(result))
        reply = ""
        while reply not in ("y", "n", "q"):
            reply = ask(QUESTION).strip().lower()
        if reply == "q":
            break
        labels[result["id"]] = reply == "y"
        labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return labels_path


def main() -> None:
    path = label(Path(sys.argv[1]))
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
