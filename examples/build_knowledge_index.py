"""Embed every knowledge chunk and save data/knowledge_index.json (decision 008).

    python -m examples.build_knowledge_index

Needs EMBEDDING_API_KEY and EMBEDDING_BASE_URL; EMBEDDING_MODEL defaults to
qwen3.7-text-embedding-flash. Rerun whenever data/knowledge/ changes: the
Agent refuses an index whose corpus hash does not match.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from eink_agent.knowledge import INDEX_PATH, KNOWLEDGE_DIR, corpus_sha256, embed_texts, embedding_text, load_chunks

DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"


def embedding_client() -> OpenAI:
    for name in ("EMBEDDING_API_KEY", "EMBEDDING_BASE_URL"):
        if not os.environ.get(name):
            raise SystemExit(f"未设置 {name}，见 .env.example。")
    return OpenAI(api_key=os.environ["EMBEDDING_API_KEY"], base_url=os.environ["EMBEDDING_BASE_URL"], timeout=60, max_retries=2)


def build_index(*, client: Any, model: str, directory: Path = KNOWLEDGE_DIR, output: Path = INDEX_PATH) -> Path:
    chunks = load_chunks(directory)
    vectors = embed_texts(client, model, [embedding_text(chunk) for chunk in chunks])
    index = {
        "model": model,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": corpus_sha256(directory),
        "dimensions": len(vectors[0]) if vectors else 0,
        "vectors": {chunk["id"]: [round(value, 6) for value in vector] for chunk, vector in zip(chunks, vectors)},
    }
    Path(output).write_text(json.dumps(index, ensure_ascii=False) + "\n", encoding="utf-8")
    return Path(output)


def main() -> None:
    model = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    with embedding_client() as client:
        path = build_index(client=client, model=model)
    index = json.loads(path.read_text(encoding="utf-8"))
    print(f"已为 {len(index['vectors'])} 个片段建立 {index['dimensions']} 维向量：{path}")


if __name__ == "__main__":
    main()
