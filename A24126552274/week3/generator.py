"""
generator.py

Generates a balanced synthetic dataset from dsa.json.
"""

import json
import random
import os

from config import *
from templates import QUERY_TEMPLATES
from responses import RESPONSE_TEMPLATES


# ============================================================
# Safe Formatter
# ============================================================

class SafeDict(dict):

    def __missing__(self, key):
        return ""


# ============================================================
# Load Dataset
# ============================================================

print("Loading dsa.json...")

with open("dsa.json", "r", encoding="utf-8") as f:

    data = json.load(f)

CONCEPTS = data["concepts"]

print(f"Loaded {len(CONCEPTS)} concepts.")


# ============================================================
# Output Folder
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Random Seed
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# Dataset
# ============================================================

dataset = []


# ============================================================
# Statistics
# ============================================================

category_counter = {}
difficulty_counter = {}
query_counter = {}
concept_counter = {}


# ============================================================
# Counter Helper
# ============================================================

def increment(counter, key):

    counter[key] = counter.get(key, 0) + 1


# ============================================================
# Example Generator
# ============================================================

def generate_examples(concept):

    examples = []

    safe = SafeDict(concept)

    for query_type, difficulty_groups in QUERY_TEMPLATES.items():

        response_templates = RESPONSE_TEMPLATES[query_type]

        for difficulty, questions in difficulty_groups.items():

            for question in questions:

                instruction = question.format_map(safe)

                for response in response_templates:

                    output = response.format_map(safe)

                    example = {

                        "instruction": instruction,
                        "input": "",
                        "output": output,

                        "concept": concept["name"],
                        "category": concept["category"],

                        "difficulty": difficulty,

                        "query_type": query_type

                    }

                    examples.append(example)

    return examples
# ============================================================
# Generate Complete Dataset
# ============================================================

print("\nGenerating examples...")

for concept in CONCEPTS:

    examples = generate_examples(concept)

    dataset.extend(examples)

    increment(concept_counter, concept["name"])


print(f"Generated {len(dataset)} raw examples.")


# ============================================================
# Remove Duplicate Examples
# ============================================================

if REMOVE_DUPLICATES:

    print("Removing duplicates...")

    seen = set()

    unique_dataset = []

    for item in dataset:

        key = (
            item["instruction"].strip(),
            item["output"].strip()
        )

        if key not in seen:

            seen.add(key)

            unique_dataset.append(item)

    removed = len(dataset) - len(unique_dataset)

    dataset = unique_dataset

    print(f"Removed {removed} duplicates.")


# ============================================================
# Shuffle Dataset
# ============================================================

if SHUFFLE_DATASET:

    print("Shuffling dataset...")

    random.shuffle(dataset)


# ============================================================
# Collect Statistics
# ============================================================

for item in dataset:

    increment(category_counter, item["category"])

    increment(difficulty_counter, item["difficulty"])

    increment(query_counter, item["query_type"])

    increment(concept_counter, item["concept"])


print("\nStatistics")

print("-------------------------------")

print("Total Examples :", len(dataset))

print("Categories     :", len(category_counter))

print("Concepts       :", len(concept_counter))

print("Difficulty     :", difficulty_counter)

print("Query Types    :", query_counter)

# ============================================================
# Save Full Dataset
# ============================================================

if EXPORT_JSON:

    print("\nSaving dataset...")

    with open(
        FULL_DATASET_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=JSON_INDENT,
            ensure_ascii=False
        )

    print(f"Dataset saved to {FULL_DATASET_FILE}")


# ============================================================
# Build Statistics
# ============================================================

statistics = {

    "total_examples": len(dataset),

    "total_concepts": len(CONCEPTS),

    "categories": category_counter,

    "difficulty": difficulty_counter,

    "query_types": query_counter,

    "concept_distribution": concept_counter

}


# ============================================================
# Save Statistics
# ============================================================

with open(
    STATISTICS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        statistics,
        f,
        indent=4,
        ensure_ascii=False
    )

print(f"Statistics saved to {STATISTICS_FILE}")


# ============================================================
# Final Report
# ============================================================

print("\n" + "=" * 60)

print("DATASET GENERATION COMPLETED")

print("=" * 60)

print(f"Concepts              : {len(CONCEPTS)}")

print(f"Examples Generated    : {len(dataset)}")

print(f"Categories            : {len(category_counter)}")

print(f"Difficulty Levels     : {len(difficulty_counter)}")

print(f"Query Types           : {len(query_counter)}")

print()

print("Output Files")

print("--------------------------")

print(FULL_DATASET_FILE)

print(STATISTICS_FILE)

print()

print("Next Step:")

print("Run splitter.py")

print("=" * 60)