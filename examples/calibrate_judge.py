"""Recompute judge calibration after labels change, without calling a model.

    python -m examples.calibrate_judge evals/reports/judge_A.json

Uses the verdicts saved in judge_A.json and the current evals/labels/A.json,
and writes calibration_A.md next to it. The judge file is not modified.
Use it after adjudicating disagreements: the judge file keeps the
calibration against the first-pass labels.
"""

import json
from pathlib import Path
import sys

from eink_agent.llm_judge import calibrate

LABELS_DIR = Path(__file__).resolve().parents[1] / "evals" / "labels"


def recalibrate(judge_path: Path, *, labels_dir: Path = LABELS_DIR) -> Path:
    judge_path = Path(judge_path)
    judged = json.loads(judge_path.read_text(encoding="utf-8"))
    labels = json.loads((Path(labels_dir) / judged["report"]).read_text(encoding="utf-8"))
    verdicts = judged["verdicts"]
    result = calibrate(verdicts, labels)
    missed = [cid for cid, v in verdicts.items() if "error" not in v and cid in labels and v["grounded"] and not labels[cid]]
    alarms = [cid for cid, v in verdicts.items() if "error" not in v and cid in labels and not v["grounded"] and labels[cid]]

    lines = [
        f"# 校准（当前标注）：{judged['report']}",
        "",
        f"裁判结果来自 `{judge_path.name}`（`{judged['judge_model']}`），标注来自 `evals/labels/{judged['report']}` 的当前版本。",
        "未重新调用模型。若标注在看到裁判结果后修改过，这里的一致率会偏高。",
        "",
        f"- 一致：{result['agreed']} / {result['compared']}",
        f"- 裁判漏判：{result['judge_missed']}（{', '.join(missed) or '无'}）",
        f"- 裁判误报：{result['judge_false_alarm']}（{', '.join(alarms) or '无'}）",
        f"- 未比较：{result['skipped']}",
    ]
    output = judge_path.with_name(judge_path.name.replace("judge_", "calibration_", 1).replace(".json", ".md"))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    output = recalibrate(Path(sys.argv[1]))
    print(output.read_text(encoding="utf-8"))
    print(f"已保存：{output}")


if __name__ == "__main__":
    main()
