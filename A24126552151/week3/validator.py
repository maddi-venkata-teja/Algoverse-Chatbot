"""
validator.py

Validates the generated dataset and writes statistics.json
"""

import json
from collections import Counter

from config import *

# ==========================================================
# Load Dataset
# ==========================================================

with open(FULL_DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} examples")

# ==========================================================
# Duplicate Check
# ==========================================================

seen = set()
duplicates = 0

for item in dataset:

    key = (
        item["instruction"],
        item["output"]
    )

    if key in seen:
        duplicates += 1

    seen.add(key)

# ==========================================================
# Statistics
# ==========================================================

category_counts = Counter(
    item["category"]
    for item in dataset
)

concept_counts = Counter(
    item["concept"]
    for item in dataset
)

difficulty_counts = Counter(
    item["difficulty"]
    for item in dataset
)

query_counts = Counter(
    item["query_type"]
    for item in dataset
)

# ==========================================================
# Balance Checker
# ==========================================================

def is_balanced(counter):

    values = list(counter.values())

    return (
        min(values) == max(values)
        if values
        else True
    )

# ==========================================================
# Statistics Dictionary
# ==========================================================

stats = {

    "total_examples": len(dataset),

    "duplicates": duplicates,

    "balanced": {

        "categories": is_balanced(category_counts),

        "concepts": is_balanced(concept_counts),

        "difficulty": is_balanced(difficulty_counts),

        "query_types": is_balanced(query_counts),

    },

    "category_distribution": dict(category_counts),

    "concept_distribution": dict(concept_counts),

    "difficulty_distribution": dict(difficulty_counts),

    "query_type_distribution": dict(query_counts)

}

# ==========================================================
# Save
# ==========================================================

with open(
    STATISTICS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        stats,
        f,
        indent=4,
        ensure_ascii=False
    )

# ==========================================================
# Print Report
# ==========================================================

print("\nValidation Report")
print("=" * 50)

print(f"Total Examples : {len(dataset)}")
print(f"Duplicates     : {duplicates}")

print("\nBalance Status")

print(
    "Categories :",
    stats["balanced"]["categories"]
)

print(
    "Concepts   :",
    stats["balanced"]["concepts"]
)

print(
    "Difficulty :",
    stats["balanced"]["difficulty"]
)

print(
    "Query Types:",
    stats["balanced"]["query_types"]
)

print("\nStatistics saved to")

print(STATISTICS_FILE)