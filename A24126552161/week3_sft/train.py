"""
Week 3 — LoRA/QLoRA Fine-Tuning

Loads a small base model in 4-bit (QLoRA), attaches a LoRA adapter,
and fine-tunes it on the Week 2 code-comment dataset.

Run this on a GPU machine (RunPod / Kaggle) — it will be extremely slow
or may fail on CPU-only laptops.
"""

import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def load_config(path="week3_sft/configs/lora_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def format_prompt(example):
    """Turn an {instruction, input, output} row into a single training string."""
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Code:\n{example['input']}\n\n"
        f"### Comment:\n{example['output']}"
    )
    return {"text": text}


def main():
    cfg = load_config()

    print(f"Loading base model: {cfg['base_model']}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    print("Loading datasets...")
    train_dataset = load_dataset("json", data_files=cfg["train_file"], split="train")
    val_dataset = load_dataset("json", data_files=cfg["val_file"], split="train")

    train_dataset = train_dataset.map(format_prompt)
    val_dataset = val_dataset.map(format_prompt)

    print(f"Train examples: {len(train_dataset)} | Val examples: {len(val_dataset)}")

    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        max_seq_length=cfg["max_seq_length"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        bf16=True,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {cfg['adapter_save_path']}")
    trainer.save_model(cfg["adapter_save_path"])
    tokenizer.save_pretrained(cfg["adapter_save_path"])

    print("Done. Adapter saved.")


if __name__ == "__main__":
    main()