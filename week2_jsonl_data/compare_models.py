"""
week2_jsonl_data/compare_models.py
─────────────────────────────────────
Week 2 — Embedding Model Comparison
Contribution by: Sushanth (A24126552152)

Benchmarks BGE-Small-v1.5, E5-Large, and MiniLM on a small set of
DSA query/passage pairs and writes results as JSONL.

Usage:
    python week2_jsonl_data/compare_models.py
"""

import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

MODELS = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
]

EVAL_PAIRS = [
    ("Explain stack applications", "A stack follows LIFO and is used in function call management, undo operations, and expression evaluation."),
    ("What is the time complexity of binary search?", "Binary search runs in O(log n) time on a sorted array."),
    ("How does a queue differ from a stack?", "A queue follows FIFO, removing elements in the order they were added."),
]

OUTPUT_JSONL = Path("week2_jsonl_data/model_comparison.jsonl")


def benchmark_model(model_name: str) -> dict:
    print(f"\n🔍 Benchmarking {model_name} ...")
    model = SentenceTransformer(model_name)

    queries = [q for q, _ in EVAL_PAIRS]
    passages = [p for _, p in EVAL_PAIRS]

    start = time.time()
    query_emb = model.encode(queries, convert_to_tensor=True)
    passage_emb = model.encode(passages, convert_to_tensor=True)
    elapsed = time.time() - start

    sims = util.cos_sim(query_emb, passage_emb).diagonal().tolist()
    avg_sim = sum(sims) / len(sims)

    return {
        "model": model_name,
        "avg_similarity": round(avg_sim, 4),
        "embedding_dim": int(query_emb.shape[1]),
        "encode_time_sec": round(elapsed, 4),
        "num_pairs_tested": len(EVAL_PAIRS),
    }


def main():
    results = [benchmark_model(m) for m in MODELS]

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Wrote benchmark results to {OUTPUT_JSONL}")
    best = max(results, key=lambda r: r["avg_similarity"])
    print(f"🏆 Best by similarity: {best['model']}")


if __name__ == "__main__":
    main()
