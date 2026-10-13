"""Tests for the component knowledge base (decision 008). Offline only:
vector search uses vectors supplied by the test, never a real embedding call.
"""

import json
from pathlib import Path

import pytest

from eink_agent.knowledge import (
    KnowledgeBase,
    load_chunks,
    parse_document,
    rrf_merge,
    tokenize,
)

CORPUS = {
    "01-kaleido.md": """title: Kaleido 彩色墨水屏
verified_at: 2026-10-13
source: E Ink 官网 | https://www.eink.com/kaleido
source: IT之家 | https://www.ithome.com/0/942/771.htm

## 原理
Kaleido 在黑白墨水屏上加一层彩色滤光片。

## 分辨率
彩色内容的分辨率是黑白的一半，例如黑白 300 PPI、彩色 150 PPI。
""",
    "02-pen.md": """title: 电磁笔
verified_at: 2026-10-13
source: Wikipedia | https://en.wikipedia.org/wiki/Graphics_tablet

## 不需要电池
电磁笔的能量来自屏幕发出的信号，所以不用充电。
""",
}


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    for name, text in CORPUS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_document_header_and_sections_are_parsed():
    doc = parse_document("01-kaleido", CORPUS["01-kaleido.md"])

    assert doc["title"] == "Kaleido 彩色墨水屏"
    assert doc["verified_at"] == "2026-10-13"
    assert doc["sources"] == [
        {"title": "E Ink 官网", "url": "https://www.eink.com/kaleido"},
        {"title": "IT之家", "url": "https://www.ithome.com/0/942/771.htm"},
    ]
    assert [heading for heading, _ in doc["sections"]] == ["原理", "分辨率"]


@pytest.mark.parametrize(
    "text",
    [
        "verified_at: 2026-10-13\nsource: a | https://a.com\n\n## x\ny\n",          # no title
        "title: t\nverified_at: 2026-10-13\n\n## x\ny\n",                          # no source
        "title: t\nverified_at: 2026-10-13\nsource: a | ftp://a.com\n\n## x\ny\n",  # bad url
        "title: t\nverified_at: 2026-10-13\nsource: a | https://a.com\n",           # no sections
    ],
)
def test_documents_without_required_metadata_are_rejected(text):
    with pytest.raises(ValueError):
        parse_document("bad", text)


def test_each_section_becomes_one_chunk_with_metadata(corpus_dir):
    chunks = load_chunks(corpus_dir)

    assert [chunk["id"] for chunk in chunks] == ["01-kaleido#1", "01-kaleido#2", "02-pen#1"]
    chunk = chunks[1]
    assert chunk["doc_id"] == "01-kaleido"
    assert chunk["title"] == "Kaleido 彩色墨水屏"
    assert chunk["section"] == "分辨率"
    assert chunk["source_url"] == "https://www.eink.com/kaleido"
    assert chunk["verified_at"] == "2026-10-13"
    assert "150 PPI" in chunk["text"]


def test_tokenize_uses_chinese_bigrams_and_lowercase_words():
    assert tokenize("彩色墨水 Kaleido 3") == ["彩色", "色墨", "墨水", "kaleido", "3"]
    assert tokenize("暖光") == ["暖光"]
    assert tokenize("光") == ["光"]


def test_keyword_search_ranks_the_matching_section_first(corpus_dir):
    kb = KnowledgeBase(load_chunks(corpus_dir))

    hits = kb.keyword_search("彩色屏的分辨率为什么低", k=3)

    assert hits[0]["id"] == "01-kaleido#2"
    assert all(hit["score"] > 0 for hit in hits)


def test_keyword_search_returns_nothing_without_overlap(corpus_dir):
    kb = KnowledgeBase(load_chunks(corpus_dir))

    assert kb.keyword_search("售后保修多久", k=3) == []


def test_vector_search_uses_cosine_and_threshold(corpus_dir):
    chunks = load_chunks(corpus_dir)
    vectors = {"01-kaleido#1": [1.0, 0.0], "01-kaleido#2": [0.8, 0.6], "02-pen#1": [0.0, 1.0]}
    kb = KnowledgeBase(chunks, vectors=vectors)

    hits = kb.vector_search([1.0, 0.0], k=2, min_score=0.5)

    assert [hit["id"] for hit in hits] == ["01-kaleido#1", "01-kaleido#2"]
    assert hits[0]["score"] == pytest.approx(1.0)
    assert kb.vector_search([0.0, -1.0], k=3, min_score=0.5) == []


def test_rrf_merge_rewards_items_ranked_by_both():
    merged = rrf_merge([["a", "b", "c"], ["b", "c", "d"]], k=60)

    assert merged[0] == "b"
    assert set(merged) == {"a", "b", "c", "d"}


def test_index_file_must_match_the_corpus(corpus_dir, tmp_path):
    chunks = load_chunks(corpus_dir)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"corpus_sha256": "stale", "vectors": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="corpus"):
        KnowledgeBase.from_files(corpus_dir, index)

    assert KnowledgeBase.from_files(corpus_dir, tmp_path / "missing.json").vectors == {}
    assert len(KnowledgeBase.from_files(corpus_dir, tmp_path / "missing.json").chunks) == len(chunks)


def test_real_corpus_loads_and_every_chunk_has_a_source():
    chunks = load_chunks(Path(__file__).resolve().parents[1] / "data" / "knowledge")

    assert len({chunk["doc_id"] for chunk in chunks}) == 20
    assert all(chunk["source_url"].startswith("https://") for chunk in chunks)
    assert all(chunk["text"].strip() for chunk in chunks)


def _embedding_client(dimension_of=lambda text: [float(len(text)), 1.0]):
    from unittest.mock import Mock

    client = Mock()

    def create(model, input):
        return Mock(data=[Mock(embedding=dimension_of(text)) for text in input])

    client.embeddings.create.side_effect = create
    return client


def test_embed_texts_batches_requests_and_keeps_order():
    from eink_agent.knowledge import embed_texts

    client = _embedding_client()
    texts = [f"t{i}" * (i + 1) for i in range(45)]

    vectors = embed_texts(client, "embed-model", texts, batch_size=20)

    assert client.embeddings.create.call_count == 3
    assert [len(call.kwargs["input"]) for call in client.embeddings.create.call_args_list] == [20, 20, 5]
    assert vectors[44] == [float(len(texts[44])), 1.0]


def test_build_index_records_model_and_corpus_hash(corpus_dir, tmp_path):
    from eink_agent.knowledge import corpus_sha256
    from examples.build_knowledge_index import build_index

    output = tmp_path / "index.json"
    build_index(client=_embedding_client(), model="embed-model", directory=corpus_dir, output=output)

    index = json.loads(output.read_text(encoding="utf-8"))
    assert index["model"] == "embed-model"
    assert index["corpus_sha256"] == corpus_sha256(corpus_dir)
    assert set(index["vectors"]) == {"01-kaleido#1", "01-kaleido#2", "02-pen#1"}
    assert KnowledgeBase.from_files(corpus_dir, output).vectors


def test_retrieval_metrics_are_document_level():
    from examples.evaluate_retrieval import precision_at_k, recall_at_k

    hits = [{"doc_id": "a"}, {"doc_id": "a"}, {"doc_id": "c"}]

    assert recall_at_k(hits, ["a", "b"]) == 0.5
    assert precision_at_k(hits, ["a", "b"], k=3) == pytest.approx(2 / 3)
    assert recall_at_k([], ["a"]) == 0.0
    assert precision_at_k([], ["a"], k=3) == 0.0


def test_retrieval_cases_file_is_well_formed():
    cases = json.loads((Path(__file__).resolve().parents[1] / "evals" / "retrieval_cases.json").read_text(encoding="utf-8"))
    doc_ids = {chunk["doc_id"] for chunk in load_chunks(Path(__file__).resolve().parents[1] / "data" / "knowledge")}

    assert len(cases) >= 30
    assert len({case["id"] for case in cases}) == len(cases)
    assert sum(not case["relevant_docs"] for case in cases) >= 5
    for case in cases:
        assert set(case["relevant_docs"]) <= doc_ids, case["id"]


def test_evaluate_retrieval_compares_three_methods(corpus_dir, tmp_path):
    from examples.build_knowledge_index import build_index
    from examples.evaluate_retrieval import run_retrieval_evaluation

    cases = [
        {"id": "q1", "question": "彩色分辨率", "relevant_docs": ["01-kaleido"]},
        {"id": "x1", "question": "售后保修", "relevant_docs": []},
    ]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    index = tmp_path / "index.json"
    client = _embedding_client()
    build_index(client=client, model="embed-model", directory=corpus_dir, output=index)

    json_path, markdown_path = run_retrieval_evaluation(
        cases_path=cases_path, client=client, model="embed-model",
        directory=corpus_dir, index_path=index, report_dir=tmp_path / "reports",
        vector_thresholds=[0.0, 0.99],
    )

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(report["methods"]) == {"keyword", "vector", "hybrid", "hybrid_gated"}
    assert "gate_sweep" in report
    keyword = report["methods"]["keyword"]
    assert keyword["recall_at_3"] == 1.0
    assert keyword["no_answer_empty_rate"] == 1.0
    assert report["run"]["answerable"] == 1 and report["run"]["unanswerable"] == 1
    assert "Recall@3" in markdown_path.read_text(encoding="utf-8")


def test_gated_hybrid_returns_nothing_when_the_best_vector_match_is_weak(corpus_dir):
    chunks = load_chunks(corpus_dir)
    vectors = {"01-kaleido#1": [1.0, 0.0], "01-kaleido#2": [0.8, 0.6], "02-pen#1": [0.0, 1.0]}
    kb = KnowledgeBase(chunks, vectors=vectors)

    # Keyword search alone would match "彩色", but the vector gate says no.
    assert kb.keyword_search("彩色", k=3)
    assert kb.gated_hybrid_search("彩色", [-1.0, -0.1], k=3, gate=0.5) == []

    hits = kb.gated_hybrid_search("彩色分辨率", [0.8, 0.6], k=3, gate=0.5)
    assert hits[0]["id"] == "01-kaleido#2"


def test_searcher_embeds_the_query_and_applies_the_gate(corpus_dir):
    from eink_agent.knowledge import KnowledgeSearcher

    chunks = load_chunks(corpus_dir)
    vectors = {"01-kaleido#1": [1.0, 0.0], "01-kaleido#2": [0.8, 0.6], "02-pen#1": [0.0, 1.0]}
    asked = []
    searcher = KnowledgeSearcher(
        KnowledgeBase(chunks, vectors=vectors),
        embed=lambda text: asked.append(text) or ([0.8, 0.6] if "分辨率" in text else [-1.0, 0.0]),
        gate=0.5,
    )

    hits = searcher.search("彩色分辨率")
    assert asked == ["彩色分辨率"]
    assert hits[0] == {
        "chunk_id": "01-kaleido#2",
        "title": "Kaleido 彩色墨水屏",
        "section": "分辨率",
        "text": chunks[1]["text"],
        "source_url": "https://www.eink.com/kaleido",
        "verified_at": "2026-10-13",
    }
    assert searcher.search("售后") == []
