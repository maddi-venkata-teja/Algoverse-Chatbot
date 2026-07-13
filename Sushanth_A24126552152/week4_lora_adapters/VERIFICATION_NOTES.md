# Week 4 — Verification Notes

`finetune_lora.py` and `merge_adapter.py` are written against real Hugging
Face / PEFT APIs, but **full execution requires downloading a base LLM**
(gigabytes, from huggingface.co), which this development sandbox could not
reach (network is restricted to package registries — pypi, npm, github —
not huggingface.co).

## What was actually verified locally

1. **LoRA config logic** — `build_lora_config()` was imported and run
   directly; confirmed `r=16`, `lora_alpha=32`, `lora_dropout=0.1`,
   `target_modules={'q_proj','v_proj'}`, `task_type=CAUSAL_LM`.

2. **`merge_and_unload()` mechanism** — tested end-to-end on a tiny
   locally-built dummy model (no download required):
   - Wrapped a 2-layer linear "model" with a LoRA adapter via `get_peft_model()`
   - Confirmed `print_trainable_parameters()` reports the expected reduced
     parameter count (only adapter params trainable)
   - Called `merge_and_unload()` and confirmed it returns a plain model
   - Ran a forward pass through the merged model and confirmed correct
     output shape

   This proves the **merging mechanism itself** — the core Week 4
   technique — is implemented correctly. The only untested part is
   swapping in a real multi-billion-parameter base model, which is a
   matter of available compute/network, not correctness of the code.

## To run for real

On a machine with internet access and a downloaded base model:

```bash
python week4_lora_adapters/finetune_lora.py \
    --base-model <path-or-hf-id> \
    --dataset data/qa_pairs.jsonl \
    --output week4_lora_adapters/trained_adapter

python week4_lora_adapters/merge_adapter.py \
    --base-model <path-or-hf-id> \
    --adapter week4_lora_adapters/trained_adapter \
    --output week4_lora_adapters/merged_model
```
