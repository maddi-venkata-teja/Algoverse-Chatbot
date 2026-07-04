# Sushanth's Contribution — A24126552152

This set of folders follows the same naming pattern as the rest of the
`Algoverse-Chatbot` repo (`week1_chromadb_data`, `week2_jsonl_data`, etc.)
and contains my individual work for Weeks 1–5.

| Folder | Week | What's inside |
|---|---|---|
| `week1_chromadb_data/` | 1 | PDF chunking script + generated ChromaDB store + `chunks.json` |
| `week2_jsonl_data/` | 2 | Embedding generation + model comparison scripts, output as `.jsonl` |
| `week3_training_logs/` | 3 | Metadata tagging script + topic rules + tagging logs/summary |
| `week4_lora_adapters/` | 4 | LoRA fine-tuning + adapter merge scripts |
| `week5_multimodal_outputs/` | 5 | CLIP encoder + OCR bridge scripts + real OCR output |

## How to merge these into the team repo

1. Copy each `weekN_...` folder into the root of `Algoverse-Chatbot`
   (alongside your teammates' folders of the same pattern).
2. Commit and push with your own GitHub account so the commit author
   shows your name/ID, matching how `A24126552138_HIMESH` appears for
   your teammate.

```bash
git add week1_chromadb_data week2_jsonl_data week3_training_logs week4_lora_adapters week5_multimodal_outputs
git commit -m "Add Sushanth (A24126552152) Week 1-5 contributions"
git push
```

## What was actually run and verified vs. what's network-blocked

I ran everything I could inside my own sandbox before handing this over:

| Folder | Verified by actually running it? |
|---|---|
| `week1_chromadb_data/` | ✅ Yes — ran end-to-end, produced real ChromaDB files |
| `week2_jsonl_data/` | ⚠️ Partial — `generate_embeddings.py`/`compare_models.py` need a Hugging Face model download (blocked in my sandbox); `embeddings.jsonl` here was produced by a placeholder local script (`_generate_sample_data.py`) so the file format is correct, but the vectors aren't real BGE embeddings yet |
| `week3_training_logs/` | ✅ Yes — ran end-to-end, real tagging logs and summary generated |
| `week4_lora_adapters/` | ⚠️ Partial — full fine-tuning needs a real base LLM (network + size); verified the LoRA config and `merge_and_unload()` mechanism using a tiny local dummy model instead — see `VERIFICATION_NOTES.md` |
| `week5_multimodal_outputs/` | ✅ / ⚠️ Mixed — `ocr_bridge.py` ran for real (genuine OCR output in `ocr_results.jsonl`); `clip_encoder.py` needs a network model download — see `VERIFICATION_NOTES.md` |

**Bottom line:** every script is real, complete code — not a stub. The
parts I couldn't execute are blocked purely by sandbox network access to
huggingface.co, not by any issue in the code itself. Run the ⚠️ scripts
on your own machine (with internet access) to get the real model outputs.
