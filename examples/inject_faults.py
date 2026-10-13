"""Build answers with known unsupported claims to measure judge misses.

    python -m examples.inject_faults evals/reports/A.json

Each fault appends one sentence to a saved answer that contradicts the tool
results of that case (wrong weight, wrong price, waterproof or stylus
support reversed or invented). Such answers are not grounded by
construction, so no hand label is needed. Output is deterministic and goes
to evals/faults/<report>.json (decision 007).

Known limitation: the appended sentence may contradict an earlier sentence
in the same answer, which can make it easier to spot than a real mistake.
"""

import json
from pathlib import Path
import sys
from typing import Any

FAULTS_DIR = Path(__file__).resolve().parents[1] / "evals" / "faults"
FAULT_TYPES = ("weight", "waterproof", "price", "stylus")
DEFAULT_LIMIT = 10


def _devices(result: dict[str, Any]) -> list[dict[str, Any]]:
    devices = []
    for call in result["tool_calls"]:
        if call["status"] != "ok" or not call.get("result"):
            continue
        items = call["result"] if isinstance(call["result"], list) else [call["result"]]
        devices += [item for item in items if isinstance(item, dict) and "model" in item]
    return devices


def _claim(fault_type: str, device: dict[str, Any]) -> str | None:
    """A sentence that contradicts this device's tool data, or None."""
    name = f"{device.get('brand', '')} {device['model']}".strip()
    if fault_type == "weight" and isinstance(device.get("weight_g"), int):
        return f"{name} 重 {device['weight_g'] + 60} 克。"
    if fault_type == "waterproof" and "is_waterproof" in device:
        return f"{name} 不防水。" if device["is_waterproof"] == 1 else f"{name} 支持 IPX8 防水。"
    if fault_type == "price" and isinstance(device.get("lowest_price_cny"), int) and device["lowest_price_cny"] > 400:
        return f"{name} 现价 {device['lowest_price_cny'] - 300} 元。"
    if fault_type == "stylus" and "supports_stylus" in device:
        return f"{name} 不支持手写笔。" if device["supports_stylus"] == 1 else f"{name} 支持手写笔。"
    return None


def generate_faults(report: dict[str, Any], limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    """Deterministically build up to ``limit`` faulty answers.

    Fault types rotate across cases so one case does not supply every fault.
    """
    cases = [(result, _devices(result)) for result in report["results"] if result["answer"]]
    cases = [(result, devices) for result, devices in cases if devices]
    faults: list[dict[str, Any]] = []

    for offset in range(len(FAULT_TYPES)):
        for index, (result, devices) in enumerate(cases):
            if len(faults) >= limit:
                return faults
            fault_type = FAULT_TYPES[(index + offset) % len(FAULT_TYPES)]
            claim = next((c for c in (_claim(fault_type, d) for d in devices) if c), None)
            if claim is None:
                continue
            faults.append(
                {
                    "id": f"{result['id']}#{fault_type}",
                    "source_case": result["id"],
                    "fault_type": fault_type,
                    "claim": claim,
                    "input": result["input"],
                    "history": result.get("history", []),
                    "tool_calls": result["tool_calls"],
                    "answer": f"{result['answer']}\n\n{claim}",
                }
            )
    return faults


def main() -> None:
    report_path = Path(sys.argv[1])
    faults = generate_faults(json.loads(report_path.read_text(encoding="utf-8")))
    FAULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = FAULTS_DIR / report_path.name
    output.write_text(json.dumps(faults, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for fault in faults:
        print(f"{fault['id']}: {fault['claim']}")
    print(f"已保存 {len(faults)} 条：{output}")


if __name__ == "__main__":
    main()
