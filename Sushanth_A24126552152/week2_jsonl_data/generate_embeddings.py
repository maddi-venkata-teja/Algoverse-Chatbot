"""
week2_jsonl_data/generate_embeddings.py
──────────────────────────────────────────
Week 2 — Embedding Model Comparison & Generation
Contribution by: Sushanth (A24126552152)

Loads the chunks produced in Week 1 and encodes them with
BGE-Small-v1.5, writing the result as JSONL (one embedded
record per line) — matching the team's week2_jsonl_data format.

Usage:
    python week2_jsonl_data/generate_embeddings.py
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("week1_chromadb_data/chunks.json")
OUTPUT_JSONL = Path("week2_jsonl_data/embeddings.jsonl")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run week1_chromadb_data/build_knowledge_base.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print(f"🔤 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    chunks = load_chunks()
    print(f"📦 Loaded {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True).tolist()
    print(f"🧮 Generated {len(vectors)} embeddings of dim {len(vectors[0]) if vectors else 0}")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for chunk, vector in zip(chunks, vectors):
            record = {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "embedding": vector,
                "embedding_model": EMBEDDING_MODEL,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(chunks)} embedded records to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
