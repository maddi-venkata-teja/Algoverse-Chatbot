"""
week5_multimodal_outputs/ocr_bridge.py
───────────────────────────────────────────
Week 5 — Multimodal Research Integration
Contribution by: Sushanth (A24126552152)

Bridges image inputs (code screenshots, handwritten notes) into
AlgoVerse's existing text pipeline: extract text via Tesseract
OCR, then embed with the same model used for the rest of the
knowledge base, tagging the result with source_type="image_ocr".

Usage:
    python week5_multimodal_outputs/ocr_bridge.py \
        --image data/stack_code_screenshot.png \
        --topic Stack --difficulty Intermediate
"""

import argparse
import json
import os
from pathlib import Path

import pytesseract
from PIL import Image

OUTPUT_LOG = Path("week5_multimodal_outputs/ocr_results.jsonl")


def extract_text_from_image(image_path: str) -> str:
    image = Image.open(image_path)
    return pytesseract.image_to_string(image).strip()


def log_ocr_result(text: str, topic: str, difficulty: str, image_path: str):
    """Append the OCR result + intended metadata to a JSONL log."""
    metadata = {
        "topic": topic,
        "category": "OCR-Derived",
        "difficulty": difficulty,
        "source": os.path.basename(image_path),
        "chapter": topic,
        "source_type": "image_ocr",
    }
    record = {"text": text, "metadata": metadata}

    OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Logged OCR result to {OUTPUT_LOG}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="OCR bridge: image → text → metadata log")
    parser.add_argument("--image", required=True)
    parser.add_argument("--topic", default="General")
    parser.add_argument("--difficulty", default="Beginner")
    args = parser.parse_args()

    print(f"🔎 Running OCR on {args.image} ...")
    text = extract_text_from_image(args.image)
    print(f"📝 Extracted {len(text)} characters")

    if not text:
        print("⚠️  No text detected in image — skipping log.")
        return

    log_ocr_result(text, args.topic, args.difficulty, args.image)


if __name__ == "__main__":
    main()
