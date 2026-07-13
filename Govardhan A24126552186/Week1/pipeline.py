import fitz  # PyMuPDF
import re
import json
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ─── STEP 1: Detect PDF type ───────────────────────────────────────
def detect_pdf_type(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    if len(text.strip()) > 50:
        return "digital"
    else:
        return "scanned"

# ─── STEP 2a: Extract text from digital PDF ────────────────────────
def extract_digital(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages

# ─── STEP 2b: Extract text from scanned PDF via OCR ───────────────
def extract_scanned(pdf_path):
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        raise ImportError(
            "Scanned PDF detected. Install OCR dependencies:\n"
            "  pip install pytesseract pdf2image Pillow\n"
            "  brew install tesseract poppler  (macOS)"
        )

    print("  → Running OCR on scanned pages (this may take a minute)...")
    images = convert_from_path(pdf_path, dpi=300)
    pages = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        pages.append({"page": i + 1, "text": text})
        print(f"     OCR done: page {i + 1}/{len(images)}")
    return pages

# ─── STEP 3: Clean text ────────────────────────────────────────────
def clean_text(raw_text):
    text = re.sub(r'-\n', '', raw_text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append('')
            continue

        if len(stripped) < 4:
            continue

        total = len(stripped)
        non_alpha = sum(1 for c in stripped if not c.isalnum() and c not in ' .,!?-:()')
        if total > 0 and (non_alpha / total) > 0.4:
            continue

        # Skip lines that look like random letter chunks
        # (real words have vowels — garbage often doesn't)
        words = stripped.split()
        if words:
            garbage_words = 0
            for word in words:
                letters_only = re.sub(r'[^a-zA-Z]', '', word)
                if len(letters_only) >= 3:
                    has_vowel = bool(re.search(r'[aeiouAEIOU]', letters_only))
                    if not has_vowel:
                        garbage_words += 1
            if len(words) > 0 and (garbage_words / len(words)) > 0.5:
                continue

        # ── NEW: Skip short mixed-case gibberish lines ──────────
        # Catches lines like "VIC Unis", "OU Slits", "AN ow) lole"
        # Pattern: 1-4 words, contains brackets OR
        # all-caps word next to lowercase word with no real sentence structure
        if len(words) <= 4:
            # Has brackets or special chars = diagram label
            if re.search(r'[()[\]{}|\\/@#]', stripped):
                continue
            # All-caps word mixed with short lowercase = diagram noise
            has_allcaps = any(w.isupper() and len(w) >= 2 for w in words)
            has_lowercase = any(w.islower() and len(w) <= 4 for w in words)
            if has_allcaps and has_lowercase and len(words) <= 3:
                continue
            # Ends with no punctuation AND starts with caps AND very short
            # e.g. "VIC Unis" or "OU Slits"
            if (len(words) <= 3
                and stripped[0].isupper()
                and not stripped.endswith(('.', '?', '!', ':'))
                and not any(c.isdigit() for c in stripped)
                and len(stripped) < 20):
                # Only skip if it doesn't look like a real heading
                # Real headings are usually 1 word or a known phrase
                real_heading_words = ['introduction', 'conclusion', 'summary',
                                      'overview', 'chapter', 'section', 'table',
                                      'figure', 'example', 'note', 'what', 'how',
                                      'why', 'when', 'where', 'select', 'insert',
                                      'update', 'delete', 'create', 'drop']
                first_word_lower = words[0].lower()
                if first_word_lower not in real_heading_words:
                    continue

        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def clean_all_pages(pages):
    return [{"page": p["page"], "text": clean_text(p["text"])} for p in pages]

# ─── STEP 4: Garbage chunk detector ───────────────────────────────
def is_garbage_chunk(text):
    lines = text.strip().split('\n')
    garbage_line_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Too short
        if len(stripped) < 4:
            garbage_line_count += 1
            continue

        # Too many symbols
        total = len(stripped)
        non_alpha = sum(1 for c in stripped if not c.isalnum() and c not in ' .,!?-:()')
        if total > 0 and (non_alpha / total) > 0.4:
            garbage_line_count += 1
            continue

        # Words with no vowels
        words = stripped.split()
        if words:
            no_vowel_words = 0
            for word in words:
                letters = re.sub(r'[^a-zA-Z]', '', word)
                if len(letters) >= 3 and not re.search(r'[aeiouAEIOU]', letters):
                    no_vowel_words += 1
            if no_vowel_words / len(words) > 0.5:
                garbage_line_count += 1

    total_lines = len([l for l in lines if l.strip()])
    if total_lines == 0:
        return True
    return (garbage_line_count / total_lines) > 0.4

# ─── STEP 4: Chunk text ────────────────────────────────────────────
def chunk_text(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    all_chunks = []
    skipped = 0

    for page in pages:
        if not page["text"].strip():
            print(f"  ⚠ Page {page['page']} is empty after cleaning, skipping.")
            continue

        chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(chunks):

            if is_garbage_chunk(chunk):
                skipped += 1
                continue

            all_chunks.append({
                "text": chunk,
                "page": page["page"],
                "chunk_id": f"page{page['page']}_chunk{i}"
            })

    if skipped > 0:
        print(f"  🗑  Skipped {skipped} garbage chunks")

    return all_chunks

# ─── STEP 5: Generate embeddings ───────────────────────────────────
def generate_embeddings(chunks):
    print("  Loading embedding model (first time takes ~1 min)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    return chunks

# ─── SAVE TO JSONL ─────────────────────────────────────────────────
def save_to_jsonl(chunks, filename="chunks.jsonl"):
    with open(filename, "w") as f:
        for chunk in chunks:
            record = {
                "chunk_id": chunk["chunk_id"],
                "page":     chunk["page"],
                "text":     chunk["text"],
            }
            f.write(json.dumps(record) + "\n")
    print(f"  ✅ Saved {len(chunks)} chunks → {filename}")

# ─── LOAD FROM JSONL ───────────────────────────────────────────────
def load_from_jsonl(filename="chunks.jsonl"):
    chunks = []
    with open(filename, "r") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"  ✅ Loaded {len(chunks)} chunks ← {filename}")
    return chunks

# ─── STEP 6: Store in ChromaDB ─────────────────────────────────────
def store_in_chromadb(chunks, collection_name="algoverse_docs"):
    if not chunks:
        print("  ✗ No chunks to store. Check that your PDF has extractable text.")
        return None

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[{"page": chunk["page"]} for chunk in chunks]
    )
    print(f"  ✅ Stored {len(chunks)} chunks → ChromaDB")
    return collection

# ─── MAIN: Run the full pipeline ───────────────────────────────────
def run_pipeline(pdf_path):
    print(f"\n🚀 Running pipeline on: {pdf_path}")

    print("\n[1/6] Detecting PDF type...")
    pdf_type = detect_pdf_type(pdf_path)
    print(f"  → {pdf_type} PDF")

    print("\n[2/6] Extracting text...")
    if pdf_type == "digital":
        pages = extract_digital(pdf_path)
    else:
        pages = extract_scanned(pdf_path)
    print(f"  → {len(pages)} pages extracted")

    print("\n[3/6] Cleaning text...")
    pages = clean_all_pages(pages)

    print("\n[4/6] Chunking...")
    chunks = chunk_text(pages)
    print(f"  → {len(chunks)} chunks created")

    if not chunks:
        print("\n✗ Pipeline stopped: 0 chunks. OCR may have failed or PDF is blank.")
        return None

    print("\n[5/6] Generating embeddings...")
    chunks = generate_embeddings(chunks)

    print("\n[6/6] Saving to JSONL + storing in ChromaDB...")
    save_to_jsonl(chunks)
    collection = store_in_chromadb(chunks)

    print("\n✓ Pipeline complete!")
    print(f"   → chunks.jsonl : {len(chunks)} chunks saved")
    print(f"   → chroma_db/   : {len(chunks)} vectors indexed")
    return collection

# ─── RUN IT ────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline("SQL-Manual.pdf")