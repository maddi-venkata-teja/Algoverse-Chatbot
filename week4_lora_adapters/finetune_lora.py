"""
week4_lora_adapters/finetune_lora.py
───────────────────────────────────────
Week 4 — Adapter Merging (training phase)
Contribution by: Sushanth (A24126552152)

Fine-tunes a base causal LLM on DSA question-answer pairs using
LoRA (Low-Rank Adaptation), training only a small fraction of
total parameters.

Usage:
    python week4_lora_adapters/finetune_lora.py \
        --base-model week4_lora_adapters/base_model \
        --dataset data/qa_pairs.jsonl \
        --output week4_lora_adapters/trained_adapter
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def build_lora_config() -> LoraConfig:
    """LoRA hyperparameters used for AlgoVerse fine-tuning."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )


def load_qa_dataset(path: Path, tokenizer) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    def tokenize(example):
        text = f"### Question:\n{example['prompt']}\n\n### Answer:\n{example['response']}"
        tokens = tokenizer(text, truncation=True, max_length=512, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = Dataset.from_list(records)
    return dataset.map(tokenize, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune AlgoVerse base model")
    parser.add_argument("--base-model", default="week4_lora_adapters/base_model")
    parser.add_argument("--dataset", default="data/qa_pairs.jsonl")
    parser.add_argument("--output", default="week4_lora_adapters/trained_adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    print(f"📦 Loading base model from {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)

    lora_config = build_lora_config()
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"⚠️  {dataset_path} not found — skipping training.")
        return

    train_dataset = load_qa_dataset(dataset_path, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(model=peft_model, args=training_args, train_dataset=train_dataset)

    print("🚀 Starting LoRA fine-tuning ...")
    trainer.train()

    peft_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"✅ Adapter saved to {args.output}")


if __name__ == "__main__":
    main()
