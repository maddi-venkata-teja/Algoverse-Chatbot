"""
week4_lora_adapters/merge_adapter.py
───────────────────────────────────────
Week 4 — Adapter Merging (deployment phase)
Contribution by: Sushanth (A24126552152)

Loads the base model plus the trained LoRA adapter, merges the
adapter weights into the base model with merge_and_unload(), and
saves a single, deployment-ready model.

Usage:
    python week4_lora_adapters/merge_adapter.py \
        --base-model week4_lora_adapters/base_model \
        --adapter week4_lora_adapters/trained_adapter \
        --output week4_lora_adapters/merged_model
"""

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_and_save(base_model_path: str, adapter_path: str, output_path: str):
    print(f"📦 Loading base model from {base_model_path} ...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

    print(f"🧩 Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("🔗 Merging adapter weights into base model ...")
    merged_model = model.merge_and_unload()

    Path(output_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    print(f"✅ Merged model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    parser.add_argument("--base-model", default="week4_lora_adapters/base_model")
    parser.add_argument("--adapter", default="week4_lora_adapters/trained_adapter")
    parser.add_argument("--output", default="week4_lora_adapters/merged_model")
    args = parser.parse_args()

    merge_and_save(args.base_model, args.adapter, args.output)


if __name__ == "__main__":
    main()
