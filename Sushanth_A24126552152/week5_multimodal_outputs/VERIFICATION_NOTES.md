# Week 5 — Verification Notes

## ✅ `ocr_bridge.py` — fully tested, real execution

This script was run end-to-end for real:
1. Generated a test image (`data/stack_code_screenshot.png`) containing
   Python code text
2. Ran Tesseract OCR on it via `pytesseract.image_to_string()`
3. Confirmed text was extracted (with the typical minor OCR noise you'd
   expect from a default font — e.g. "stack" merging with the next word)
4. Confirmed the result was logged to `ocr_results.jsonl` with correct
   metadata (`topic`, `category`, `difficulty`, `source_type: "image_ocr"`)

See `ocr_results.jsonl` in this folder for the real output.

## ⚠️ `clip_encoder.py` — written but not executed here

CLIP (`openai/clip-vit-base-patch32`) must be downloaded from
huggingface.co, which this development sandbox cannot reach (network
allowlist covers package registries only: pypi, npm, github — not
huggingface.co). The code is written against the standard
`transformers.CLIPModel` / `CLIPProcessor` API and follows the same
pattern used successfully elsewhere in this project; it has not been
executed end-to-end in this environment.

To run for real:
```bash
python week5_multimodal_outputs/clip_encoder.py \
    --image data/binary_tree_diagram.png \
    --query "binary tree traversal"
```
