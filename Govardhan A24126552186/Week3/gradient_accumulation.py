import json
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── CONFIG ─────────────────────────────────────────────────────────
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small, CPU-friendly cross-encoder
CHUNKS_FILE = "chunks.jsonl"
OUTPUT_DIR = "./reranker_model"

REAL_BATCH_SIZE = 4          # what actually fits in memory at once
ACCUMULATION_STEPS = 8       # how many real batches to add up
EFFECTIVE_BATCH_SIZE = REAL_BATCH_SIZE * ACCUMULATION_STEPS   # = 32
EPOCHS = 3
LEARNING_RATE = 2e-5


# ─── STEP 1: Build training pairs from your chunks.jsonl ───────────
# We don't have human-labeled (query, chunk, relevance) data, so we
# synthesize training pairs directly from your chunks:
#   POSITIVE pair -> a pseudo-question generated FROM a chunk, paired
#                     with THAT SAME chunk (label = 1.0, relevant)
#   NEGATIVE pair -> the same pseudo-question, paired with a RANDOM
#                     chunk from a different page (label = 0.0, irrelevant)
def make_pseudo_query(chunk_text):
    """Turn the first sentence of a chunk into a rough 'question' stand-in.
    This is a simple heuristic — good enough to teach the reranker the
    shape of (question, relevant-passage) vs (question, random-passage)."""
    first_sentence = chunk_text.strip().split(". ")[0]
    return first_sentence[:120]


def build_training_pairs(chunks_file):
    with open(chunks_file, "r") as f:
        chunks = [json.loads(line) for line in f]

    if len(chunks) < 4:
        raise ValueError("Need at least 4 chunks in chunks.jsonl to build training pairs.")

    examples = []
    for chunk in chunks:
        query = make_pseudo_query(chunk["text"])

        # Positive example
        examples.append({"query": query, "passage": chunk["text"], "label": 1.0})

        # Negative example — pick a random OTHER chunk (different page if possible)
        other_chunks = [c for c in chunks if c["chunk_id"] != chunk["chunk_id"]]
        negative = random.choice(other_chunks)
        examples.append({"query": query, "passage": negative["text"], "label": 0.0})

    random.shuffle(examples)
    print(f"  Built {len(examples)} training pairs ({len(examples)//2} positive, {len(examples)//2} negative)")
    return examples


class RerankerDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["query"],
            ex["passage"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex["label"], dtype=torch.float)
        return item


# ─── STEP 2: Training loop WITH gradient accumulation ───────────────
def train_reranker():
    print("\n[1/4] Loading base cross-encoder model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  → device: {device}")

    print("\n[2/4] Building training pairs from chunks.jsonl...")
    examples = build_training_pairs(CHUNKS_FILE)
    dataset = RerankerDataset(examples, tokenizer)
    loader = DataLoader(dataset, batch_size=REAL_BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    print(f"\n[3/4] Training with gradient accumulation")
    print(f"  real batch size      = {REAL_BATCH_SIZE}")
    print(f"  accumulation steps   = {ACCUMULATION_STEPS}")
    print(f"  effective batch size = {EFFECTIVE_BATCH_SIZE}")

    model.train()
    step_count = 0
    start = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            loss = loss_fn(logits, labels)
            loss = loss / ACCUMULATION_STEPS   # scale down before accumulating
            loss.backward()                    # gradients ADD UP across steps
            epoch_loss += loss.item() * ACCUMULATION_STEPS

            # Only step the optimizer every ACCUMULATION_STEPS mini-batches
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

        # flush any leftover gradients at the end of the epoch
        if (i + 1) % ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1}/{EPOCHS} — avg loss: {avg_loss:.4f}")

    elapsed = time.time() - start
    print(f"\n  → Training done in {elapsed:.1f}s, {step_count} optimizer steps total")

    print("\n[4/4] Saving fine-tuned reranker...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  ✅ Saved reranker → {OUTPUT_DIR}")
    return model, tokenizer


# ─── STEP 3: Use the reranker to re-score retrieved chunks ──────────
def rerank(query, candidate_chunks, model=None, tokenizer=None, top_k=3):
    """Takes chunks already retrieved by the bi-encoder (query.py) and
    re-scores them with the cross-encoder for higher precision."""
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR, num_labels=1)
        model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    scored = []
    with torch.no_grad():
        for chunk_text in candidate_chunks:
            enc = tokenizer(query, chunk_text, truncation=True, padding=True,
                             max_length=256, return_tensors="pt").to(device)
            score = model(**enc).logits.item()
            scored.append((score, chunk_text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    train_reranker()