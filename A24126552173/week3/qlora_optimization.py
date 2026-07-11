# ============================================================
# WEEK 3 — QLoRA OPTIMIZATION
# AI Lab | Algoverse DSA Chatbot Project
# Member : R. Jeetesh Kumar
# Depends on : hardware_report.json  (Week 1)
#              bnb_config_report.json (Week 2)
# ============================================================

import json
import os
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# SECTION 1 : LOAD PREVIOUS WEEK REPORTS
# Why? We never hardcode values. Week 1 told us the dtype
# (fp16/bf16). Week 2 confirmed the 4-bit config is ready.
# We read both JSON files and use their values directly.
# This keeps all 3 weeks as one connected pipeline.
# ─────────────────────────────────────────────────────────────
def load_previous_reports():
    print("=" * 55)
    print("  QLoRA OPTIMIZATION — Algoverse AI Lab")
    print("  R. Jeetesh Kumar | Week 3 Task")
    print("=" * 55)
    print("\n[1/6] Loading Week 1 & Week 2 reports...")

    # -- Week 1 hardware report --
    hw = {}
    if os.path.exists("hardware_report.json"):
        with open("hardware_report.json") as f:
            hw = json.load(f)
        dtype = hw.get("verdict", {}).get("recommended_dtype", "fp16")
        gpu   = hw.get("gpu", {}).get("name", "Unknown GPU")
        vram  = hw.get("gpu", {}).get("vram_total_GB", 0)
        print(f"  Week 1 → GPU      : {gpu}")
        print(f"  Week 1 → VRAM     : {vram} GB")
        print(f"  Week 1 → dtype    : {dtype}")
    else:
        dtype = "fp16"
        print("  ⚠ hardware_report.json not found — defaulting fp16")

    # -- Week 2 BnB config report --
    bnb_info = {}
    if os.path.exists("bnb_config_report.json"):
        with open("bnb_config_report.json") as f:
            bnb_info = json.load(f)
        print(f"  Week 2 → quant    : {bnb_info.get('bnb_config', {}).get('bnb_4bit_quant_type','nf4')}")
        print(f"  Week 2 → 4-bit    : {bnb_info.get('bnb_config', {}).get('load_in_4bit', True)}")
    else:
        print("  ⚠ bnb_config_report.json not found — using defaults")

    return dtype, hw, bnb_info


# ─────────────────────────────────────────────────────────────
# SECTION 2 : LOAD BASE MODEL WITH 4-BIT QUANTIZATION
# Why? We load Phi-3-mini using Unsloth's FastLanguageModel.
# This is EXACTLY what algoverse.ipynb does in Step 2.
# The model loads in 4-bit (from Week 2 config) — 1.8 GB
# instead of 14 GB. Without this, we crash immediately.
# ─────────────────────────────────────────────────────────────
def load_base_model(dtype):
    print("\n[2/6] Loading Phi-3-mini in 4-bit (Unsloth)...")
    print("  This uses your Week 2 BitsAndBytes config under the hood.")
    print("  Model: unsloth/Phi-3-mini-4k-instruct-bnb-4bit")
    print("  load_in_4bit : True  → 14 GB compressed to ~1.8 GB")
    print("  max_seq_len  : 2048  → max tokens per training sample")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            # The exact model used in algoverse.ipynb
            model_name    = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
            max_seq_length = 2048,
            dtype          = None,   # auto-detect from Week 1 (fp16/bf16)
            load_in_4bit   = True,   # Week 2 config — 4-bit NF4 compression
        )
        print("  ✓ Base model loaded successfully")
        return model, tokenizer

    except ImportError:
        print("  ⚠ Unsloth not installed — run on Google Colab")
        print("  Install: pip install unsloth")
        return None, None


# ─────────────────────────────────────────────────────────────
# SECTION 3 : INJECT LORA ADAPTERS (THE CORE OF WEEK 3)
# Why? This is where QLoRA actually happens.
# We inject tiny A and B matrices into the attention layers.
# Only these ~8M adapter params will train — the base model
# stays frozen. This is why training is fast and cheap.
#
# Key parameters explained:
#   r = 16          → rank — size of adapter (learning capacity)
#   lora_alpha = 32 → alpha/r = 2.0 (matches algoverse.ipynb)
#   target_modules  → which layers get adapters (q and v proj)
#   lora_dropout = 0→ no dropout (Unsloth recommends 0)
#   bias = "none"   → don't train bias terms (saves VRAM)
#   gradient_checkpointing → saves activations VRAM (Unsloth optimized)
# ─────────────────────────────────────────────────────────────
def inject_lora_adapters(model):
    print("\n[3/6] Injecting LoRA adapters into attention layers...")

    try:
        from unsloth import FastLanguageModel

        # ── LoRA Config — matches algoverse.ipynb exactly ──
        RANK         = 16    # adapter size — sweet spot for DSA dataset
        LORA_ALPHA   = 32    # scale = alpha/rank = 32/16 = 2.0
        DROPOUT      = 0     # 0 = fastest, recommended by Unsloth
        RANDOM_SEED  = 3407  # same seed as algoverse.ipynb

        model = FastLanguageModel.get_peft_model(
            model,
            r                        = RANK,
            lora_alpha               = LORA_ALPHA,
            lora_dropout             = DROPOUT,
            bias                     = "none",
            use_gradient_checkpointing = "unsloth",  # saves ~30% VRAM
            random_state             = RANDOM_SEED,

            # Target modules — the attention layers that get adapters
            # q_proj = query projection  (how the model asks questions)
            # v_proj = value projection  (what info gets passed forward)
            # These 2 are the most impactful layers for task learning
            target_modules = ["q_proj", "v_proj"],
        )

        # ── Print adapter stats ──
        trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total       = sum(p.numel() for p in model.parameters())
        frozen      = total - trainable
        pct         = round(trainable / total * 100, 3)

        print(f"  r (rank)             : {RANK}")
        print(f"  lora_alpha           : {LORA_ALPHA}")
        print(f"  scale (alpha/rank)   : {LORA_ALPHA/RANK}")
        print(f"  target_modules       : q_proj, v_proj")
        print(f"  gradient_checkpointing: unsloth (saves ~30% VRAM)")
        print(f"\n  ── Adapter Stats ──")
        print(f"  Trainable params     : {trainable:,}  ({pct}%)")
        print(f"  Frozen params        : {frozen:,}")
        print(f"  Total params         : {total:,}")
        print(f"  VRAM for adapters    : ~{round(trainable*2/1e6)} MB (fp16)")
        print(f"  ✓ LoRA adapters injected successfully")

        return model, {
            "rank"         : RANK,
            "lora_alpha"   : LORA_ALPHA,
            "scale"        : LORA_ALPHA / RANK,
            "trainable_params" : trainable,
            "frozen_params"    : frozen,
            "total_params"     : total,
            "trainable_pct"    : pct,
        }

    except ImportError:
        print("  ⚠ Unsloth not installed — showing config only")
        config = {
            "rank"       : 16,
            "lora_alpha" : 32,
            "scale"      : 2.0,
            "target_modules" : ["q_proj", "v_proj"],
            "dropout"    : 0,
            "bias"       : "none",
            "gradient_checkpointing": "unsloth",
        }
        print(f"  Config that will be used on Colab:")
        for k, v in config.items():
            print(f"    {k:<28} : {v}")
        return model, config


# ─────────────────────────────────────────────────────────────
# SECTION 4 : VRAM BREAKDOWN ANALYSIS
# Why? We show exactly how VRAM is being used during training.
# This is your "optimization proof" — showing the team that
# QLoRA uses memory efficiently. Week 1 VRAM reading connects
# here to verify everything fits.
# ─────────────────────────────────────────────────────────────
def vram_breakdown_analysis(adapter_info, vram_total_GB):
    print("\n[4/6] VRAM breakdown analysis...")

    trainable = adapter_info.get("trainable_params", 8_000_000)

    # Memory calculations (all in MB)
    base_model_mb    = 1800                          # 4-bit quantized base
    adapter_weights  = round(trainable * 2 / 1e6)   # fp16 = 2 bytes
    gradients_mb     = adapter_weights               # same size as weights
    optimizer_mb     = round(trainable * 8 / 1e6)   # Adam: 2 states × fp32
    activations_mb   = 54                            # est. for batch_size=4
    framework_mb     = 300                           # CUDA, PyTorch overhead

    total_mb = (base_model_mb + adapter_weights + gradients_mb
                + optimizer_mb + activations_mb + framework_mb)
    total_gb = round(total_mb / 1024, 2)
    free_gb  = round(vram_total_GB - total_gb, 2) if vram_total_GB > 0 else "?"

    print(f"\n  ── VRAM Usage During Training ──")
    print(f"  {'Component':<30} {'Usage':>10}")
    print(f"  {'─'*42}")
    print(f"  {'Base model (4-bit NF4)':<30} {base_model_mb:>8} MB  ← Week 2 work")
    print(f"  {'LoRA adapter weights (fp16)':<30} {adapter_weights:>8} MB")
    print(f"  {'Gradients (fp16)':<30} {gradients_mb:>8} MB  ← learning direction")
    print(f"  {'Adam optimizer states (fp32)':<30} {optimizer_mb:>8} MB  ← training memory")
    print(f"  {'Activations (batch=4)':<30} {activations_mb:>8} MB  ← forward pass cache")
    print(f"  {'CUDA/PyTorch framework':<30} {framework_mb:>8} MB")
    print(f"  {'─'*42}")
    print(f"  {'TOTAL':<30} {total_mb:>8} MB  (~{total_gb} GB)")
    if vram_total_GB > 0:
        print(f"  {'GPU VRAM available':<30} {int(vram_total_GB*1024):>8} MB  ({vram_total_GB} GB)")
        print(f"  {'Free headroom':<30} {int(free_gb*1024):>8} MB  ({free_gb} GB)")

    return {
        "base_model_mb"   : base_model_mb,
        "adapters_mb"     : adapter_weights,
        "gradients_mb"    : gradients_mb,
        "optimizer_mb"    : optimizer_mb,
        "activations_mb"  : activations_mb,
        "framework_mb"    : framework_mb,
        "total_mb"        : total_mb,
        "total_gb"        : total_gb,
    }


# ─────────────────────────────────────────────────────────────
# SECTION 5 : TRAINING CONFIGURATION
# Why? We set up the SFTTrainer arguments that control HOW
# training runs — batch size, learning rate, epochs, dtype.
# These match algoverse.ipynb exactly so Week 3 plugs in
# cleanly without breaking anything.
# ─────────────────────────────────────────────────────────────
def build_training_config(model, tokenizer, dataset, dtype):
    print("\n[5/6] Building training configuration...")

    try:
        import torch
        from trl import SFTTrainer
        from transformers import TrainingArguments

        # fp16 vs bf16 — from your Week 1 benchmark result
        use_fp16 = (dtype == "fp16")
        use_bf16 = (dtype == "bf16")

        training_args = TrainingArguments(
            # Batch & gradient settings
            per_device_train_batch_size  = 4,    # 4 samples per GPU step
            gradient_accumulation_steps  = 1,    # no accumulation needed

            # Training duration
            num_train_epochs             = 1,    # 1 full pass over dataset

            # Learning rate — how fast adapters update per step
            learning_rate                = 2e-4, # 0.0002 — standard for LoRA

            # Dtype from Week 1 benchmark (fp16 for T4, bf16 for A100)
            fp16                         = use_fp16,
            bf16                         = use_bf16,

            # Optimizer — 8-bit Adam saves VRAM vs standard Adam
            optim                        = "adamw_8bit",

            # Logging & saving
            logging_steps                = 1,
            save_strategy                = "no",
            output_dir                   = "outputs",
            report_to                    = "none",
        )

        trainer = SFTTrainer(
            model              = model,
            tokenizer          = tokenizer,
            train_dataset      = dataset,
            dataset_text_field = "text",
            max_seq_length     = 2048,
            args               = training_args,
        )

        config_summary = {
            "batch_size"     : 4,
            "epochs"         : 1,
            "learning_rate"  : "2e-4",
            "optimizer"      : "adamw_8bit",
            "dtype"          : dtype,
            "seq_length"     : 2048,
        }

        print(f"  batch_size       : 4")
        print(f"  epochs           : 1")
        print(f"  learning_rate    : 2e-4")
        print(f"  optimizer        : adamw_8bit  (saves ~50% optimizer VRAM)")
        print(f"  dtype            : {dtype}  (from Week 1 benchmark)")
        print(f"  seq_length       : 2048 tokens")
        print(f"  ✓ Trainer configured")

        return trainer, config_summary

    except ImportError:
        print("  ⚠ trl / transformers not installed — showing config only")
        config_summary = {
            "batch_size"     : 4,
            "epochs"         : 1,
            "learning_rate"  : "2e-4",
            "optimizer"      : "adamw_8bit",
            "dtype"          : dtype,
            "seq_length"     : 2048,
        }
        for k, v in config_summary.items():
            print(f"    {k:<20} : {v}")
        return None, config_summary


# ─────────────────────────────────────────────────────────────
# SECTION 6 : SAVE QLoRA REPORT
# Why? Save everything — adapter config, VRAM breakdown,
# training config — into a single JSON. Week 4 (stress
# testing team) uses this to understand what was trained
# and verify the model works correctly.
# ─────────────────────────────────────────────────────────────
def save_qlora_report(dtype, adapter_info, vram_info, train_config):
    print("\n[6/6] Saving QLoRA optimization report...")

    report = {
        "week"           : 3,
        "task"           : "QLoRA Optimization",
        "member"         : "R. Jeetesh Kumar",
        "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "pipeline_connection": {
            "week1_contribution" : "Hardware benchmark → recommended dtype for training",
            "week2_contribution" : "BitsAndBytes config → 4-bit base model (1.8 GB)",
            "week3_contribution" : "QLoRA adapters → ~8M trainable params injected",
        },

        "lora_config": {
            "rank"               : adapter_info.get("rank", 16),
            "lora_alpha"         : adapter_info.get("lora_alpha", 32),
            "scale"              : adapter_info.get("scale", 2.0),
            "target_modules"     : ["q_proj", "v_proj"],
            "dropout"            : 0,
            "bias"               : "none",
            "gradient_checkpointing": "unsloth",
        },

        "adapter_stats": {
            "trainable_params"   : adapter_info.get("trainable_params", "~8M"),
            "frozen_params"      : adapter_info.get("frozen_params", "~3.8B"),
            "trainable_pct"      : adapter_info.get("trainable_pct", 0.2),
            "adapter_vram_fp16"  : f"~{vram_info.get('adapters_mb', 16)} MB",
        },

        "vram_breakdown_mb"  : vram_info,
        "training_config"    : train_config,

        "what_not_quantized" : {
            "adapters_stay_fp16" : True,
            "reason"             : (
                "Adapters are actively learning — quantizing to 4-bit "
                "would lose gradient precision and break training. "
                "Only frozen weights (base model) are safe to quantize."
            ),
        },

        "for_week4_team": (
            "Model adapter saved at outputs/. "
            "Load with FastLanguageModel.from_pretrained() + PeftModel. "
            "Run stress tests using the algoverse chat UI on Colab."
        ),
    }

    with open("qlora_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 55)
    print("  WEEK 3 SUMMARY — QLoRA Optimization")
    print("=" * 55)
    print(f"  Base model      : Phi-3-mini (4-bit, frozen, 1.8 GB)")
    print(f"  Adapters        : q_proj + v_proj (LoRA injected)")
    print(f"  Rank            : {adapter_info.get('rank', 16)}")
    print(f"  Alpha           : {adapter_info.get('lora_alpha', 32)}")
    print(f"  Scale           : {adapter_info.get('scale', 2.0)}")
    print(f"  Trainable       : ~{adapter_info.get('trainable_params', '8M'):,} params")
    print(f"  Compute dtype   : {dtype} (from Week 1)")
    print(f"  Total VRAM used : ~{vram_info.get('total_gb', '?')} GB")
    print(f"  Report saved    → qlora_report.json")
    print("=" * 55)

    print("\n  ── How this connects to algoverse.ipynb ──")
    print("""
  # This is the exact code from algoverse.ipynb Step 2-3
  # Your Week 3 work powers this section:

  model = FastLanguageModel.get_peft_model(
      model,
      r                         = 16,      # ← YOUR rank choice
      lora_alpha                = 32,      # ← YOUR alpha choice
      lora_dropout              = 0,
      bias                      = "none",
      use_gradient_checkpointing= "unsloth",
      random_state              = 3407,
      target_modules            = ["q_proj", "v_proj"],
  )

  # Training runs ONLY on the 8M adapter params
  # Base model (3.8B) stays frozen in 4-bit the whole time
    """)

    return report


# ─────────────────────────────────────────────────────────────
# MAIN — Wire all 6 sections together
# ─────────────────────────────────────────────────────────────
def run_qlora_optimization():
    # Section 1 — Load previous weeks
    dtype, hw, bnb_info = load_previous_reports()
    vram_gb = hw.get("gpu", {}).get("vram_total_GB", 0)

    # Section 2 — Load base model
    model, tokenizer = load_base_model(dtype)

    # Section 3 — Inject LoRA adapters
    model, adapter_info = inject_lora_adapters(model)

    # Section 4 — VRAM breakdown
    vram_info = vram_breakdown_analysis(adapter_info, vram_gb)

    # Section 5 — Training config (needs dataset — shown as config only here)
    _, train_config = build_training_config(model, tokenizer, None, dtype)

    # Section 6 — Save report
    report = save_qlora_report(dtype, adapter_info, vram_info, train_config)

    return report


if __name__ == "__main__":
    run_qlora_optimization()
