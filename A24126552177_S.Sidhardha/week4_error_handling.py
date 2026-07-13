"""
Week 4 Task: Error Handling
Student: S.SIDDARADHA | A24126552177

Error handling means catching, logging, and gracefully recovering from
failures in the AI pipeline — model loading errors, inference failures,
bad inputs, API errors, and more.
"""

import os
import json
import logging
import traceback
from functools import wraps
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ── Logging Setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("pipeline_errors.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# ── 1. Decorator: Catch & Log Any Function Error ──────────────────────────────

def handle_errors(default_return=None):
    """Decorator that catches exceptions, logs them, and returns a safe default."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"❌ Error in '{func.__name__}': {e}")
                logger.debug(traceback.format_exc())
                return default_return
        return wrapper
    return decorator


# ── 2. Safe Model Loading ─────────────────────────────────────────────────────

@handle_errors(default_return=None)
def load_model(model_path):
    if not model_path:
        raise ValueError("model_path cannot be empty.")
    if not os.path.exists(model_path) and "/" not in model_path:
        raise FileNotFoundError(f"Local model not found: {model_path}")

    logger.info(f"📦 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    logger.info("✅ Model loaded successfully.")
    return model, tokenizer


# ── 3. Safe Inference ─────────────────────────────────────────────────────────

@handle_errors(default_return={"response": None, "error": "Inference failed"})
def run_inference(pipe, prompt, max_new_tokens=200):
    if not prompt or not isinstance(prompt, str):
        raise ValueError(f"Invalid prompt: {repr(prompt)}")
    if len(prompt.strip()) == 0:
        raise ValueError("Prompt cannot be blank.")

    logger.info(f"🤖 Running inference for prompt: '{prompt[:60]}...'")
    output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    result = output[0]["generated_text"]
    logger.info("✅ Inference successful.")
    return {"response": result, "error": None}


# ── 4. Safe Dataset Loading ───────────────────────────────────────────────────

@handle_errors(default_return=[])
def load_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = []
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Skipping malformed JSON at line {line_num}: {e}")

    logger.info(f"✅ Loaded {len(data)} records from {path}")
    return data


# ── 5. Safe Checkpoint Loading ────────────────────────────────────────────────

@handle_errors(default_return=None)
def load_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints = sorted(
        [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1])
    )
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in directory.")

    latest = os.path.join(checkpoint_dir, checkpoints[-1])
    logger.info(f"🔁 Resuming from checkpoint: {latest}")
    return latest


# ── 6. Input Validation ───────────────────────────────────────────────────────

def validate_input(data: dict) -> dict:
    """Validate a single dataset record and return cleaned version or raise."""
    required_keys = ["instruction", "output"]
    errors = []

    for key in required_keys:
        if key not in data:
            errors.append(f"Missing key: '{key}'")
        elif not isinstance(data[key], str):
            errors.append(f"'{key}' must be a string, got {type(data[key]).__name__}")
        elif len(data[key].strip()) == 0:
            errors.append(f"'{key}' cannot be empty.")

    if errors:
        raise ValueError(f"Validation failed: {'; '.join(errors)}")

    return {
        "instruction": data["instruction"].strip(),
        "input": data.get("input", "").strip(),
        "output": data["output"].strip(),
    }


@handle_errors(default_return=[])
def validate_dataset(records):
    valid, invalid = [], 0
    for i, record in enumerate(records):
        try:
            valid.append(validate_input(record))
        except ValueError as e:
            logger.warning(f"⚠️ Record {i} skipped: {e}")
            invalid += 1

    logger.info(f"✅ Valid: {len(valid)} | ❌ Skipped: {invalid}")
    return valid


# ── 7. Error Report ───────────────────────────────────────────────────────────

def generate_error_report(log_file="pipeline_errors.log", report_file="error_report.json"):
    if not os.path.exists(log_file):
        logger.warning("No log file found.")
        return

    errors, warnings = [], []
    with open(log_file, 'r') as f:
        for line in f:
            if "ERROR" in line:
                errors.append(line.strip())
            elif "WARNING" in line:
                warnings.append(line.strip())

    report = {
        "total_errors": len(errors),
        "total_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"📋 Error report saved → {report_file}")
    print(f"\n📊 Summary: {len(errors)} errors, {len(warnings)} warnings")
    return report


# ── Main: Demo Pipeline with Full Error Handling ──────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Week 4: Error Handling Demo Pipeline")
    print("=" * 60)

    # Test 1: Load a valid dataset
    print("\n[1] Loading dataset...")
    records = load_jsonl("dataset_shuffled.jsonl")

    # Test 2: Validate dataset records
    print("\n[2] Validating records...")
    valid_records = validate_dataset(records)

    # Test 3: Simulate bad inputs
    print("\n[3] Testing bad inputs...")
    bad_inputs = [
        {},                          # missing keys
        {"instruction": "", "output": "ok"},  # empty instruction
        {"instruction": 123, "output": "ok"}, # wrong type
    ]
    for bad in bad_inputs:
        try:
            validate_input(bad)
        except ValueError as e:
            logger.error(f"Caught expected error: {e}")

    # Test 4: Load a non-existent checkpoint (expected failure)
    print("\n[4] Testing checkpoint loading...")
    ckpt = load_checkpoint("./checkpoints")
    if ckpt:
        print(f"   Checkpoint found: {ckpt}")
    else:
        print("   No checkpoint found (expected if Week 3 not run yet).")

    # Test 5: Load a bad model path (expected failure)
    print("\n[5] Testing bad model path...")
    result = load_model("")
    if result is None:
        print("   Gracefully handled bad model path ✅")

    # Generate error report
    print("\n[6] Generating error report...")
    generate_error_report()

    print("\n✅ Week 4 Error Handling demo complete!")
    print("   Check 'pipeline_errors.log' and 'error_report.json' for details.")
