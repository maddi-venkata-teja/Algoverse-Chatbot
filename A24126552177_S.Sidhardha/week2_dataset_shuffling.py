"""
Week 2 Task: Dataset Shuffling
Student: S.SIDDARADHA | A24...177
"""

import json
import random
from datasets import load_dataset


# ── 1. Basic JSONL Shuffle ────────────────────────────────────────────────────

def shuffle_jsonl(input_path, output_path, seed=42):
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    random.seed(seed)
    random.shuffle(data)

    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    print(f"Shuffled {len(data)} samples → {output_path}")


# ── 2. Shuffle using HuggingFace Datasets ────────────────────────────────────

def shuffle_hf(input_path, output_path, seed=42):
    dataset = load_dataset("json", data_files=input_path, split="train")
    dataset = dataset.shuffle(seed=seed)
    dataset.to_json(output_path)
    print(f"HF Shuffled {len(dataset)} samples → {output_path}")


# ── 3. Shuffle + Train/Val Split ─────────────────────────────────────────────

def shuffle_and_split(input_path, train_path, val_path, val_size=0.1, seed=42):
    dataset = load_dataset("json", data_files=input_path, split="train")
    dataset = dataset.shuffle(seed=seed)

    split = dataset.train_test_split(test_size=val_size, seed=seed)
    split["train"].to_json(train_path)
    split["test"].to_json(val_path)

    print(f"Train: {len(split['train'])} samples → {train_path}")
    print(f"Val  : {len(split['test'])} samples → {val_path}")


# ── 4. Sanity Check ──────────────────────────────────────────────────────────

def sanity_check(path, n=3):
    with open(path, 'r') as f:
        lines = [json.loads(line) for line in f if line.strip()]
    print(f"\nFirst {n} samples from {path}:")
    for i, sample in enumerate(lines[:n]):
        print(f"  [{i}] {sample}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT  = "dataset.jsonl"
    OUTPUT = "dataset_shuffled.jsonl"
    TRAIN  = "train.jsonl"
    VAL    = "val.jsonl"

    # Method 1: Basic shuffle
    shuffle_jsonl(INPUT, OUTPUT)

    # Method 2: HuggingFace shuffle
    # shuffle_hf(INPUT, OUTPUT)

    # Method 3: Shuffle + split
    # shuffle_and_split(INPUT, TRAIN, VAL)

    # Sanity check
    sanity_check(OUTPUT)
