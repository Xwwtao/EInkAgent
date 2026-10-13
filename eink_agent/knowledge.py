"""Component knowledge base: parsing, chunking and retrieval (decision 008).

Documents in data/knowledge/*.md start with header lines (title,
verified_at, one or more "source: title | url"), followed by "## " sections.
Each section is one chunk. Retrieval is keyword (BM25 over Chinese bigrams),
vector (cosine over precomputed embeddings) or hybrid (reciprocal rank
fusion). Everything here is pure Python and deterministic; embeddings are
computed elsewhere and passed in.
"""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import re
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_DIR / "data" / "knowledge"
INDEX_PATH = PROJECT_DIR / "data" / "knowledge_index.json"

# Standard BM25 defaults; the corpus is too small to tune them on.
BM25_K1 = 1.5
BM25_B = 0.75
# Standard reciprocal rank fusion constant.
RRF_K = 60

CJK_RUN = re.compile(r"[一-鿿]+")
TOKEN_PATTERN = re.compile(r"[一-鿿]+|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Chinese runs become overlapping two-character tokens; other words are
    lowercased. Bigrams match two-character terms such as 暖光 or 防水."""
    tokens: list[str] = []
    for piece in TOKEN_PATTERN.findall(text.lower()):
        if CJK_RUN.fullmatch(piece):
            tokens += [piece] if len(piece) == 1 else [piece[i:i + 2] for i in range(len(piece) - 1)]
        else:
            tokens.append(piece)
    return tokens


def parse_document(doc_id: str, text: str) -> dict[str, Any]:
    """Parse one knowledge file; missing metadata is an error, not a default."""
    header, _, body = text.partition("\n## ")
    fields: dict[str, Any] = {"sources": []}
    for line in header.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "source":
            title, _, url = value.partition("|")
            if not url.strip().startswith("https://"):
                raise ValueError(f"{doc_id}: source must be an https link: {value!r}")
            fields["sources"].append({"title": title.strip(), "url": url.strip()})
        elif key in ("title", "verified_at"):
            fields[key] = value

    if not fields.get("title") or not fields.get("verified_at") or not fields["sources"]:
        raise ValueError(f"{doc_id}: title, verified_at and at least one source are required")

    sections = []
    for block in ("## " + body).split("\n## ") if body else []:
        heading, _, content = block.removeprefix("## ").partition("\n")
        if heading.strip() and content.strip():
            sections.append((heading.strip(), " ".join(content.split())))
    if not sections:
        raise ValueError(f"{doc_id}: at least one '## ' section is required")

    return {"doc_id": doc_id, **fields, "sections": sections}


def load_chunks(directory: Path = KNOWLEDGE_DIR) -> list[dict[str, Any]]:
    """One chunk per section, in file order, each carrying its metadata."""
    chunks = []
    for path in sorted(Path(directory).glob("*.md")):
        doc = parse_document(path.stem, path.read_text(encoding="utf-8"))
        for number, (section, text) in enumerate(doc["sections"], start=1):
            chunks.append(
                {
                    "id": f"{doc['doc_id']}#{number}",
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "section": section,
                    "text": text,
                    "source_url": doc["sources"][0]["url"],
                    "sources": doc["sources"],
                    "verified_at": doc["verified_at"],
                }
            )
    return chunks


def corpus_sha256(directory: Path = KNOWLEDGE_DIR) -> str:
    """Hash of every knowledge file, so a stale vector index is detected."""
    digest = hashlib.sha256()
    for path in sorted(Path(directory).glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def embedding_text(chunk: dict[str, Any]) -> str:
    """Text sent to the embedding model: document title gives context."""
    return f"{chunk['title']}｜{chunk['section']}\n{chunk['text']}"


def rrf_merge(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion of several ranked id lists."""
    scores: Counter[str] = Counter()
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1 / (k + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class KnowledgeBase:
    """Chunks plus the keyword index and, when available, chunk vectors."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        vectors: dict[str, list[float]] | None = None,
        embedding_model: str | None = None,
    ):
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.by_id = {chunk["id"]: chunk for chunk in chunks}
        self.vectors = vectors or {}
        self._tokens = [tokenize(f"{chunk['title']} {chunk['section']} {chunk['text']}") for chunk in chunks]
        self._average_length = sum(map(len, self._tokens)) / len(self._tokens) if chunks else 0
        document_frequency = Counter(token for tokens in self._tokens for token in set(tokens))
        total = len(chunks)
        self._idf = {
            token: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for token, count in document_frequency.items()
        }

    @classmethod
    def from_files(cls, directory: Path = KNOWLEDGE_DIR, index_path: Path = INDEX_PATH) -> "KnowledgeBase":
        """Load the corpus and, if present, a vector index built from it."""
        chunks = load_chunks(directory)
        if not Path(index_path).exists():
            return cls(chunks)
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        if index.get("corpus_sha256") != corpus_sha256(directory):
            raise ValueError("Vector index does not match the knowledge corpus; rebuild it.")
        return cls(chunks, vectors=index["vectors"], embedding_model=index.get("model"))

    def _hit(self, chunk_id: str, score: float) -> dict[str, Any]:
        return {**self.by_id[chunk_id], "score": round(score, 4)}

    def keyword_search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """BM25 ranking; chunks sharing no token with the query are dropped."""
        query_tokens = tokenize(query)
        scored = []
        for chunk, tokens in zip(self.chunks, self._tokens):
            counts = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if frequency:
                    norm = BM25_K1 * (1 - BM25_B + BM25_B * len(tokens) / self._average_length)
                    score += self._idf[token] * frequency * (BM25_K1 + 1) / (frequency + norm)
            if score > 0:
                scored.append((chunk["id"], score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [self._hit(chunk_id, score) for chunk_id, score in scored[:k]]

    def vector_search(self, query_vector: list[float], k: int = 3, min_score: float = 0.0) -> list[dict[str, Any]]:
        """Cosine ranking over chunks that have vectors; below min_score is dropped."""
        scored = [
            (chunk_id, _cosine(query_vector, vector))
            for chunk_id, vector in self.vectors.items()
            if chunk_id in self.by_id
        ]
        scored = [pair for pair in scored if pair[1] >= min_score]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [self._hit(chunk_id, score) for chunk_id, score in scored[:k]]

    def hybrid_search(
        self, query: str, query_vector: list[float], k: int = 3, min_vector_score: float = 0.0, pool: int = 10
    ) -> list[dict[str, Any]]:
        """Fuse keyword and vector rankings; empty if neither finds anything."""
        keyword = [hit["id"] for hit in self.keyword_search(query, k=pool)]
        vector = [hit["id"] for hit in self.vector_search(query_vector, k=pool, min_score=min_vector_score)]
        merged = rrf_merge([ranking for ranking in (keyword, vector) if ranking])
        scores = {item: score for item, score in zip(merged, range(len(merged), 0, -1))}
        return [self._hit(chunk_id, scores[chunk_id]) for chunk_id in merged[:k]]

    def gated_hybrid_search(
        self, query: str, query_vector: list[float], k: int = 3, gate: float = 0.0
    ) -> list[dict[str, Any]]:
        """Hybrid ranking, but only when the best vector match reaches ``gate``.

        Keyword overlap alone cannot tell that the corpus has no answer
        (common bigrams such as 多久 always match something); the vector
        score can, so it decides whether to answer and the fused ranking
        decides what to return (evals/reports/retrieval_20261013T132938Z).
        """
        best = self.vector_search(query_vector, k=1)
        if not best or best[0]["score"] < gate:
            return []
        return self.hybrid_search(query, query_vector, k=k)


# DashScope accepts at most 20 texts per embedding request for this model.
EMBEDDING_BATCH_SIZE = 20


def embed_texts(client: Any, model: str, texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    """Embed texts in order through an OpenAI-compatible client."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(model=model, input=texts[start:start + batch_size])
        vectors += [item.embedding for item in response.data]
    return vectors


# Chosen on evals/reports/retrieval_20261013T133030Z (decision 008): every
# unanswerable question returned nothing, two answerable ones were lost.
KNOWLEDGE_GATE = 0.55
KNOWLEDGE_TOP_K = 3

HIT_FIELDS = ("title", "section", "text", "source_url", "verified_at")


class KnowledgeSearcher:
    """What the Agent's search_knowledge tool calls.

    ``embed`` turns the query into a vector; it wraps the network client,
    which the caller creates and passes in.
    """

    def __init__(self, kb: KnowledgeBase, embed: Any, gate: float = KNOWLEDGE_GATE, k: int = KNOWLEDGE_TOP_K):
        self.kb = kb
        self.embed = embed
        self.gate = gate
        self.k = k

    def search(self, query: str) -> list[dict[str, Any]]:
        hits = self.kb.gated_hybrid_search(query, self.embed(query), k=self.k, gate=self.gate)
        return [{"chunk_id": hit["id"], **{name: hit[name] for name in HIT_FIELDS}} for hit in hits]
