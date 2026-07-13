"""
week1_chromadb_data/build_knowledge_base.py
─────────────────────────────────────────────
Week 1 — Project Setup & Knowledge Base Preparation
Contribution by: Sushanth (A24126552152)

Extracts text from source PDFs in data/, splits it into semantic
chunks, and stores them as a ChromaDB-ready collection under
week1_chromadb_data/chroma_store/.

Usage:
    python week1_chromadb_data/build_knowledge_base.py
"""

import json
from pathlib import Path

import chromadb
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
CHUNKS_OUT = Path("week1_chromadb_data/chunks.json")
CHROMA_DB_PATH = "week1_chromadb_data/chroma_store"
COLLECTION_NAME = "algoverse_week1_chunks"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 40


def extract_text(pdf_path: Path) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)


def build_chunks(data_dir: Path = DATA_DIR) -> list[dict]:
    records = []
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️  No PDFs found in {data_dir}/. Add source documents and re-run.")
        return records

    for pdf_path in pdf_files:
        print(f"📄 Processing {pdf_path.name} ...")
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            records.append({
                "source": pdf_path.name,
                "chunk_id": f"{pdf_path.stem}_{i:04d}",
                "text": chunk.strip(),
            })
        print(f"   → {len(chunks)} chunks")
    return records


def store_in_chromadb(records: list[dict]):
    """
    Store raw chunks in ChromaDB. We pass placeholder embeddings here
    (all zeros) because real embeddings are generated in Week 2 —
    this avoids ChromaDB silently downloading its default embedding
    model, and keeps Week 1's responsibility scoped to chunking/storage.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if records:
        placeholder_dim = 384  # matches BGE-Small-v1.5 used in Week 2
        placeholder_embeddings = [[0.0] * placeholder_dim for _ in records]
        collection.add(
            ids=[r["chunk_id"] for r in records],
            embeddings=placeholder_embeddings,
            documents=[r["text"] for r in records],
            metadatas=[{"source": r["source"]} for r in records],
        )
    print(f"✅ Stored {len(records)} chunks in ChromaDB at '{CHROMA_DB_PATH}'")


def main():
    records = build_chunks()

    CHUNKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(records)} chunks to {CHUNKS_OUT}")

    store_in_chromadb(records)


if __name__ == "__main__":
    main()
