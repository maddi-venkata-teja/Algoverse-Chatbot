"""
Week 3 Task: Checkpoint Saving
Student: S.SIDDARADHA | A24126552177

Checkpoint saving means saving the model's weights at regular intervals
during fine-tuning so that training can be resumed if it crashes,
and the best-performing version can be recovered.
"""

import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType


# ── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # lightweight model for testing
DATASET_FILE    = "dataset_shuffled.jsonl"               # output from Week 2
OUTPUT_DIR      = "./checkpoints"
CHECKPOINT_STEPS = 50    # save a checkpoint every 50 steps
MAX_STEPS       = 200    # total training steps (adjust as needed)
SEED            = 42


# ── 1. Load Tokenizer & Model ────────────────────────────────────────────────

def load_model_and_tokenizer(model_name):
    print(f"📦 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # required for causal LM

    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


# ── 2. Apply LoRA (lightweight fine-tuning adapter) ──────────────────────────

def apply_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── 3. Prepare Dataset ───────────────────────────────────────────────────────

def prepare_dataset(tokenizer, dataset_file, max_length=256):
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    def tokenize(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        tokens = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(tokenize, remove_columns=dataset.column_names)


# ── 4. Training Arguments with Checkpoint Saving ────────────────────────────

def get_training_args(output_dir, checkpoint_steps, max_steps):
    return TrainingArguments(
        output_dir=output_dir,

        # Checkpoint saving config
        save_strategy="steps",          # save at fixed step intervals
        save_steps=checkpoint_steps,    # save every N steps
        save_total_limit=3,             # keep only last 3 checkpoints (saves disk)
        load_best_model_at_end=True,    # auto-load best checkpoint when done
        metric_for_best_model="loss",
        greater_is_better=False,

        # Training config
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,                     # set True if GPU supports it
        logging_steps=10,
        report_to="none",               # set "wandb" if using Weights & Biases
        seed=SEED,
    )


# ── 5. Resume from Checkpoint ────────────────────────────────────────────────

def get_last_checkpoint(output_dir):
    """Returns path to latest checkpoint if it exists, else None."""
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
    print(f"🔁 Resuming from checkpoint: {latest}")
    return latest


# ── 6. Save Checkpoint Info ──────────────────────────────────────────────────

def save_checkpoint_log(output_dir):
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    log = {
        "total_checkpoints": len(checkpoints),
        "checkpoints": sorted(checkpoints),
        "output_dir": output_dir,
    }
    log_path = os.path.join(output_dir, "checkpoint_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n📋 Checkpoint log saved → {log_path}")
    print(json.dumps(log, indent=2))


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # 2. Apply LoRA
    model = apply_lora(model)

    # 3. Prepare dataset
    dataset = prepare_dataset(tokenizer, DATASET_FILE)

    # 4. Training arguments
    training_args = get_training_args(OUTPUT_DIR, CHECKPOINT_STEPS, MAX_STEPS)

    # 5. Check for existing checkpoint (resume if found)
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    # 7. Train (resumes from checkpoint automatically if found)
    print("\n🚀 Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 8. Save final model
    final_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Final model saved → {final_path}")

    # 9. Log checkpoint summary
    save_checkpoint_log(OUTPUT_DIR)
