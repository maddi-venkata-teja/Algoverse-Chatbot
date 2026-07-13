"""
week2_jsonl_data/_generate_sample_data.py
─────────────────────────────────────────────
Helper script — NOT part of the graded pipeline.

generate_embeddings.py (the real Week 2 deliverable) requires
downloading BAAI/bge-small-en-v1.5 from Hugging Face, which needs
network access. This helper produces a *sample* embeddings.jsonl
with deterministic, locally-computed pseudo-embeddings, purely so
the repository ships with example output data to inspect/test
downstream code (retriever, tagger) without requiring a model
download.

Run generate_embeddings.py instead of this file whenever you have
internet access and want real BGE embeddings.
"""

import hashlib
import json
from pathlib import Path

CHUNKS_PATH = Path("week1_chromadb_data/chunks.json")
OUTPUT_JSONL = Path("week2_jsonl_data/embeddings.jsonl")
EMBEDDING_DIM = 384  # matches BGE-Small-v1.5 output size


def pseudo_embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Deterministically hash text into a fixed-size float vector.
    NOT a real semantic embedding — sample/placeholder data only.
    """
    vector = []
    seed = text.encode("utf-8")
    for i in range(dim):
        h = hashlib.sha256(seed + i.to_bytes(4, "little")).digest()
        # map first 4 bytes of the hash to a float in [-1, 1]
        val = int.from_bytes(h[:4], "little") / 2**32
        vector.append(round(val * 2 - 1, 6))
    return vector


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found. Run week1 build script first.")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "embedding": pseudo_embed(chunk["text"]),
                "embedding_model": "SAMPLE-PLACEHOLDER (run generate_embeddings.py for real BGE vectors)",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(chunks)} sample (placeholder) embedded records to {OUTPUT_JSONL}")
    print("⚠️  These are NOT real semantic embeddings — run generate_embeddings.py with internet access for the real pipeline.")


if __name__ == "__main__":
    main()
