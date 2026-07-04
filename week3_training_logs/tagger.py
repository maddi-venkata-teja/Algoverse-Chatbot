"""
week3_training_logs/tagger.py
─────────────────────────────────
Week 3 — Metadata Tagging
Contribution by: Sushanth (A24126552152)

Reads the chunked knowledge base (Week 1), infers structured
metadata for each chunk, and writes a tagging log capturing what
was tagged and why — useful for auditing classifier accuracy.

Usage:
    python week3_training_logs/tagger.py
"""

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from topic_rules import infer_metadata

CHUNKS_PATH = Path("week1_chromadb_data/chunks.json")
TAGGED_OUT = Path("week3_training_logs/chunks_tagged.json")
LOG_OUT = Path("week3_training_logs/tagging_log.jsonl")
SUMMARY_OUT = Path("week3_training_logs/tagging_summary.json")


def tag_chunks(chunks: list[dict]) -> list[dict]:
    tagged = []
    for chunk in chunks:
        meta = infer_metadata(chunk["text"], chunk["source"])
        tagged.append({**chunk, "metadata": meta})
    return tagged


def write_log(tagged_chunks: list[dict]):
    """Append one JSONL line per tagged chunk — a simple audit trail."""
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_OUT, "w", encoding="utf-8") as f:
        for c in tagged_chunks:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chunk_id": c["chunk_id"],
                "source": c["source"],
                "assigned_topic": c["metadata"]["topic"],
                "assigned_category": c["metadata"]["category"],
                "assigned_difficulty": c["metadata"]["difficulty"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✅ Wrote tagging log to {LOG_OUT}")


def write_summary(tagged_chunks: list[dict]):
    """Aggregate counts by topic/difficulty for a quick sanity check."""
    topic_counts = Counter(c["metadata"]["topic"] for c in tagged_chunks)
    difficulty_counts = Counter(c["metadata"]["difficulty"] for c in tagged_chunks)

    summary = {
        "total_chunks_tagged": len(tagged_chunks),
        "topic_distribution": dict(topic_counts),
        "difficulty_distribution": dict(difficulty_counts),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote tagging summary to {SUMMARY_OUT}")
    print(json.dumps(summary, indent=2))


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found. Run week1's build script first.")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"🏷️  Tagging {len(chunks)} chunks ...")
    tagged = tag_chunks(chunks)

    TAGGED_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TAGGED_OUT, "w", encoding="utf-8") as f:
        json.dump(tagged, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved tagged chunks to {TAGGED_OUT}")

    write_log(tagged)
    write_summary(tagged)


if __name__ == "__main__":
    main()
