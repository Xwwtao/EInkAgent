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
