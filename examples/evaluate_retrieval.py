"""Compare keyword, vector and hybrid retrieval on evals/retrieval_cases.json.

    python -m examples.evaluate_retrieval

Writes evals/reports/retrieval_<UTC time>.json and .md (decision 008).
Relevance is per document: a case lists the documents that answer it, or
none when the corpus has no answer.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
from typing import Any

from eink_agent.knowledge import INDEX_PATH, KNOWLEDGE_DIR, KnowledgeBase, embed_texts
from examples.build_knowledge_index import DEFAULT_EMBEDDING_MODEL, embedding_client

PROJECT_DIR = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_DIR / "evals" / "retrieval_cases.json"
REPORT_DIR = PROJECT_DIR / "evals" / "reports"
TOP_K = 3
VECTOR_THRESHOLDS = [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]


def recall_at_k(hits: list[dict[str, Any]], relevant: list[str]) -> float:
    """Share of relevant documents that appear among the hits."""
    if not relevant:
        return 0.0
    return len({hit["doc_id"] for hit in hits} & set(relevant)) / len(relevant)


def precision_at_k(hits: list[dict[str, Any]], relevant: list[str], k: int = TOP_K) -> float:
    """Share of the k slots filled by chunks from relevant documents."""
    return sum(hit["doc_id"] in relevant for hit in hits[:k]) / k


def _score(per_case: dict[str, list[dict[str, Any]]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [case for case in cases if case["relevant_docs"]]
    unanswerable = [case for case in cases if not case["relevant_docs"]]
    return {
        "recall_at_3": round(statistics.mean(recall_at_k(per_case[c["id"]], c["relevant_docs"]) for c in answerable), 4) if answerable else None,
        "precision_at_3": round(statistics.mean(precision_at_k(per_case[c["id"]], c["relevant_docs"]) for c in answerable), 4) if answerable else None,
        "no_answer_empty_rate": round(statistics.mean(not per_case[c["id"]] for c in unanswerable), 4) if unanswerable else None,
        "cases": {case_id: [hit["id"] for hit in hits] for case_id, hits in per_case.items()},
    }


def _balanced(metrics: dict[str, Any]) -> float:
    """Pick a vector threshold: answer what can be answered, return nothing otherwise."""
    return (metrics["recall_at_3"] or 0) + (metrics["no_answer_empty_rate"] or 0)


def _markdown(report: dict[str, Any]) -> str:
    run, methods = report["run"], report["methods"]
    lines = [
        f"# 检索评测 {run['started_at']}",
        "",
        f"- 用例：{run['answerable']} 个有答案、{run['unanswerable']} 个语料中没有答案（`{run['cases_file']}`）",
        f"- 语料：{run['chunks']} 个片段；向量模型 `{run['embedding_model']}`；取前 {TOP_K} 个片段；单次运行",
        f"- 向量阈值（按 Recall@3 + 无答案返回空的比例选出）：{run['vector_threshold']}；hybrid_gated 的门槛：{run['gate']}",
        "- hybrid_gated：最相关片段的向量分数低于门槛时返回空，否则用混合排名",
        "",
        "| 方法 | Recall@3 | Precision@3 | 无答案时返回空 |",
        "| --- | --- | --- | --- |",
        *[f"| {name} | {m['recall_at_3']} | {m['precision_at_3']} | {m['no_answer_empty_rate']} |" for name, m in methods.items()],
        "",
        "## 向量阈值扫描",
        "",
        "| 阈值 | Recall@3 | Precision@3 | 无答案时返回空 |",
        "| --- | --- | --- | --- |",
        *[f"| {t} | {m['recall_at_3']} | {m['precision_at_3']} | {m['no_answer_empty_rate']} |" for t, m in report["vector_sweep"].items()],
        "",
        "## hybrid_gated 门槛扫描",
        "",
        "| 门槛 | Recall@3 | Precision@3 | 无答案时返回空 |",
        "| --- | --- | --- | --- |",
        *[f"| {t} | {m['recall_at_3']} | {m['precision_at_3']} | {m['no_answer_empty_rate']} |" for t, m in report["gate_sweep"].items()],
        "",
        "阈值和门槛都是在同一组用例上选出的，结果偏乐观。",
    ]
    return "\n".join(lines) + "\n"


def run_retrieval_evaluation(
    *,
    cases_path: Path = CASES_PATH,
    client: Any,
    model: str,
    directory: Path = KNOWLEDGE_DIR,
    index_path: Path = INDEX_PATH,
    report_dir: Path = REPORT_DIR,
    vector_thresholds: list[float] = VECTOR_THRESHOLDS,
) -> tuple[Path, Path]:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    kb = KnowledgeBase.from_files(directory, index_path)
    started_at = datetime.now(timezone.utc)
    query_vectors = dict(zip([c["id"] for c in cases], embed_texts(client, model, [c["question"] for c in cases])))

    keyword = _score({c["id"]: kb.keyword_search(c["question"], TOP_K) for c in cases}, cases)
    sweep = {
        str(t): _score({c["id"]: kb.vector_search(query_vectors[c["id"]], TOP_K, min_score=t) for c in cases}, cases)
        for t in vector_thresholds
    }
    best = max(vector_thresholds, key=lambda t: (_balanced(sweep[str(t)]), -t))
    hybrid = _score(
        {c["id"]: kb.hybrid_search(c["question"], query_vectors[c["id"]], TOP_K, min_vector_score=best) for c in cases},
        cases,
    )
    gate_sweep = {
        str(t): _score({c["id"]: kb.gated_hybrid_search(c["question"], query_vectors[c["id"]], TOP_K, gate=t) for c in cases}, cases)
        for t in vector_thresholds
    }
    best_gate = max(vector_thresholds, key=lambda t: (_balanced(gate_sweep[str(t)]), -t))

    report = {
        "run": {
            "started_at": started_at.isoformat(),
            "cases_file": Path(cases_path).name,
            "answerable": sum(bool(c["relevant_docs"]) for c in cases),
            "unanswerable": sum(not c["relevant_docs"] for c in cases),
            "chunks": len(kb.chunks),
            "embedding_model": model,
            "vector_threshold": best,
            "gate": best_gate,
        },
        "methods": {"keyword": keyword, "vector": sweep[str(best)], "hybrid": hybrid, "hybrid_gated": gate_sweep[str(best_gate)]},
        "vector_sweep": {t: {k: v for k, v in m.items() if k != "cases"} for t, m in sweep.items()},
        "gate_sweep": {t: {k: v for k, v in m.items() if k != "cases"} for t, m in gate_sweep.items()},
    }
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    stem = "retrieval_" + started_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = Path(report_dir) / f"{stem}.json"
    markdown_path = Path(report_dir) / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    with embedding_client() as client:
        _, markdown_path = run_retrieval_evaluation(client=client, model=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"已保存：{markdown_path}")


if __name__ == "__main__":
    main()
