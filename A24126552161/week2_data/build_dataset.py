"""
Week 2 — Build the Code-Comment Pair Dataset

Loads CodeXGLUE's Code-to-Text (Python) dataset, filters out low-quality
pairs, reformats into instruction-tuning format, and saves train/val/test
splits as .jsonl files ready for Week 3's fine-tuning step.
"""
import re
import json
import random
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_SIZE = 3000          # total pairs to keep after filtering (keep this small for a fast Week 3)
MIN_COMMENT_WORDS = 4        # drop very short/trivial comments
MIN_CODE_CHARS = 30          # drop trivially short code snippets
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def is_good_pair(example):
    """Basic quality filter: drop trivial or malformed pairs."""
    code = example.get("code", "") or ""
    comment = example.get("docstring", "") or ""

    if len(code.strip()) < MIN_CODE_CHARS:
        return False
    if len(comment.strip().split()) < MIN_COMMENT_WORDS:
        return False
    # drop auto-generated / placeholder-looking comments
    if comment.strip().lower().startswith(("todo", "fixme", "xxx")):
        return False
    return True


def strip_docstring(code, docstring):
    """Remove the existing docstring from a code snippet so the model can't
    just copy it back out — it has to actually learn to generate one."""
    if docstring and docstring.strip() in code:
        code = code.replace(docstring, "", 1)
    code = re.sub(r'"""\s*"""', "", code)
    code = re.sub(r"'''\s*'''", "", code)
    code = re.sub(r"\n\s*\n\s*\n", "\n\n", code)
    return code.strip()


def format_example(example):
    """Convert a raw CodeXGLUE example into instruction-tuning format."""
    code_no_docstring = strip_docstring(example["code"], example["docstring"])
    return {
        "instruction": "Write a concise docstring describing what this function does.",
        "input": code_no_docstring,
        "output": example["docstring"].strip(),
    }


def main():
    print("Loading CodeXGLUE code-to-text (python)...")
    ds = load_dataset("google/code_x_glue_ct_code_to_text", "python", split="train")

    print(f"Raw examples: {len(ds)}")

    print("Filtering...")
    filtered = ds.filter(is_good_pair)
    print(f"After filtering: {len(filtered)}")

    # shuffle and take a manageable subset
    filtered = filtered.shuffle(seed=RANDOM_SEED)
    subset = filtered.select(range(min(TARGET_SIZE, len(filtered))))
    print(f"Using subset of size: {len(subset)}")

    formatted = [format_example(ex) for ex in subset]
    formatted = [
        row for row in formatted
        if len(row["input"]) > MIN_CODE_CHARS and row["output"] not in row["input"]
    ]
    print(f"After docstring-stripping cleanup: {len(formatted)}")

    # 80/10/10 split
    n = len(formatted)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_data = formatted[:train_end]
    val_data = formatted[train_end:val_end]
    test_data = formatted[val_end:]

    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} rows to {path}")

    save_jsonl(train_data, "week2_data/train.jsonl")
    save_jsonl(val_data, "week2_data/val.jsonl")
    save_jsonl(test_data, "week2_data/test.jsonl")

    # also save the full combined set for convenience
    save_jsonl(formatted, "week2_data/code_comment_pairs.jsonl")

    print("\nSample entry:")
    print(json.dumps(formatted[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()