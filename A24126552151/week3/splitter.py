"""
splitter.py

Splits the generated dataset into
Train / Validation / Test JSONL files.
"""

import json
import random

from config import *

# ==========================================================
# Load Dataset
# ==========================================================

with open(FULL_DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} examples")

# ==========================================================
# Shuffle
# ==========================================================

random.seed(RANDOM_SEED)
random.shuffle(dataset)

# ==========================================================
# Split Sizes
# ==========================================================

total = len(dataset)

train_size = int(total * TRAIN_RATIO)
validation_size = int(total * VALIDATION_RATIO)

train_data = dataset[:train_size]

validation_data = dataset[
    train_size:
    train_size + validation_size
]

test_data = dataset[
    train_size + validation_size:
]

# ==========================================================
# JSONL Writer
# ==========================================================

def write_jsonl(filename, data):

    with open(filename, "w", encoding="utf-8") as f:

        for item in data:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

# ==========================================================
# Save Files
# ==========================================================

write_jsonl(TRAIN_FILE, train_data)
write_jsonl(VALIDATION_FILE, validation_data)
write_jsonl(TEST_FILE, test_data)

# ==========================================================
# Summary
# ==========================================================

print("\nDataset Split Completed")
print("-" * 40)

print(f"Training   : {len(train_data)}")
print(f"Validation : {len(validation_data)}")
print(f"Testing    : {len(test_data)}")

print("\nFiles Saved")
print(TRAIN_FILE)
print(VALIDATION_FILE)
print(TEST_FILE)