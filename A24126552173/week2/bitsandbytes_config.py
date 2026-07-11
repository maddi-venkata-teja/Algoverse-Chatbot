# ============================================================
# WEEK 2 — BITSANDBYTES CONFIGURATION
# AI Lab | Algoverse DSA Chatbot Project
# Member : R. Jeetesh Kumar
# Depends on : hardware_report.json (Week 1 output)
# ============================================================

import json
import os
import sys

# ─────────────────────────────────────────────────────────────
# SECTION 1 : LOAD WEEK 1 HARDWARE REPORT
# Why? We don't hardcode the dtype — we READ it from the
# benchmark we ran in Week 1. This keeps both weeks connected.
# If Week 1 said fp16, we use fp16. If bf16, we use bf16.
# ─────────────────────────────────────────────────────────────
def load_hardware_report(path="hardware_report.json"):
    print("=" * 55)
    print("  BITSANDBYTES CONFIG — Algoverse AI Lab")
    print("  R. Jeetesh Kumar | Week 2 Task")
    print("=" * 55)

    print("\n[1/5] Loading Week 1 hardware report...")

    if not os.path.exists(path):
        print(f"  ⚠ {path} not found — defaulting to fp16")
        print("  (Run hardware_benchmark.py on Colab first for best results)")
        return {
            "recommended_dtype": "fp16",
            "gpu_name"         : "Unknown",
            "vram_total_GB"    : 0,
            "bf16_supported"   : False,
        }

    with open(path, "r") as f:
        report = json.load(f)

    gpu     = report.get("gpu", {})
    verdict = report.get("verdict", {})

    dtype    = verdict.get("recommended_dtype", "fp16")
    gpu_name = gpu.get("name", "Unknown GPU")
    vram     = gpu.get("vram_total_GB", 0)
    bf16     = gpu.get("bf16_supported", False)

    print(f"  GPU           : {gpu_name}")
    print(f"  VRAM          : {vram} GB")
    print(f"  bf16 support  : {bf16}")
    print(f"  Chosen dtype  : {dtype}  ← from Week 1 benchmark")

    return {
        "recommended_dtype": dtype,
        "gpu_name"         : gpu_name,
        "vram_total_GB"    : vram,
        "bf16_supported"   : bf16,
    }


# ─────────────────────────────────────────────────────────────
# SECTION 2 : MAP DTYPE STRING → torch dtype
# Why? BitsAndBytesConfig needs an actual torch.dtype object,
# not a plain string. This function does that conversion safely.
# ─────────────────────────────────────────────────────────────
def resolve_torch_dtype(dtype_str):
    print("\n[2/5] Resolving torch dtype...")

    try:
        import torch

        dtype_map = {
            "bf16" : torch.bfloat16,
            "fp16" : torch.float16,
            "fp32" : torch.float32,
        }

        dtype = dtype_map.get(dtype_str, torch.float16)
        print(f"  dtype string  : '{dtype_str}'")
        print(f"  torch dtype   : {dtype}")
        return dtype

    except ImportError:
        print("  ⚠ PyTorch not installed — returning dtype string only")
        return dtype_str


# ─────────────────────────────────────────────────────────────
# SECTION 3 : BUILD THE BITSANDBYTES CONFIG
# Why? This is the core of Week 2. BitsAndBytesConfig tells
# the model loader exactly HOW to compress the weights:
#
#   load_in_4bit          → compress from fp32 to 4-bit
#   bnb_4bit_quant_type   → NF4 = best quality 4-bit format
#   bnb_4bit_use_double_quant → quantize the quantization
#                              constants too (saves ~0.4 GB extra)
#   bnb_4bit_compute_dtype → which dtype to use during actual
#                            math (fp16 or bf16 from Week 1)
# ─────────────────────────────────────────────────────────────
def build_bnb_config(compute_dtype):
    print("\n[3/5] Building BitsAndBytesConfig...")

    try:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit             = True,        # compress weights to 4-bit
            bnb_4bit_quant_type      = "nf4",       # NF4 = NormalFloat4, best quality
            bnb_4bit_use_double_quant = True,       # double quantization saves ~0.4 GB
            bnb_4bit_compute_dtype   = compute_dtype, # math runs in fp16/bf16
        )

        print("  load_in_4bit             : True")
        print("  bnb_4bit_quant_type      : nf4  (NormalFloat4)")
        print("  bnb_4bit_use_double_quant: True")
        print(f"  bnb_4bit_compute_dtype   : {compute_dtype}")
        print("  ✓ BitsAndBytesConfig created successfully")

        return bnb_config

    except ImportError:
        # transformers not installed — return config as a plain dict
        # (useful when running on a machine without the full ML stack)
        print("  ⚠ transformers not installed — returning config as dict")
        config_dict = {
            "load_in_4bit"              : True,
            "bnb_4bit_quant_type"       : "nf4",
            "bnb_4bit_use_double_quant" : True,
            "bnb_4bit_compute_dtype"    : str(compute_dtype),
        }
        return config_dict


# ─────────────────────────────────────────────────────────────
# SECTION 4 : ESTIMATE MEMORY SAVINGS
# Why? We show the team exactly how much VRAM is saved by
# using 4-bit quantization vs the original fp32 weights.
# This proves WHY BitsAndBytes is necessary for Colab.
# ─────────────────────────────────────────────────────────────
def estimate_memory_savings(vram_total_GB):
    print("\n[4/5] Estimating memory savings...")

    # Phi-3-mini has 3.8 billion parameters
    params_billion  = 3.8
    bytes_per_param = {
        "fp32" : 4.0,
        "fp16" : 2.0,
        "bf16" : 2.0,
        "int8" : 1.0,
        "int4" : 0.5,   # ← what BitsAndBytes gives us
    }

    savings = {}
    for fmt, bpp in bytes_per_param.items():
        gb = (params_billion * 1e9 * bpp) / (1024 ** 3)
        fits = (vram_total_GB > 0) and (gb <= vram_total_GB)
        savings[fmt] = {
            "bytes_per_param" : bpp,
            "vram_needed_GB"  : round(gb, 2),
            "fits_on_gpu"     : fits if vram_total_GB > 0 else "unknown",
        }

    print(f"\n  Phi-3-mini ({params_billion}B parameters)")
    print(f"  {'Format':<8} {'VRAM needed':>12} {'Fits on GPU':>14}")
    print(f"  {'─'*8} {'─'*12} {'─'*14}")
    for fmt, info in savings.items():
        fits_str = "✓" if info['fits_on_gpu'] == True else (
                   "✗" if info['fits_on_gpu'] == False else "?")
        marker = "  ← BitsAndBytes" if fmt == "int4" else ""
        print(f"  {fmt:<8} {info['vram_needed_GB']:>10.2f} GB {fits_str:>12}{marker}")

    return savings


# ─────────────────────────────────────────────────────────────
# SECTION 5 : SAVE CONFIG REPORT + SHOW FINAL SUMMARY
# Why? Save the full config as a JSON so Week 3 (fine-tuning)
# team members can directly use these settings in training.
# ─────────────────────────────────────────────────────────────
def save_and_summarise(hw_info, bnb_config, memory_savings):
    print("\n[5/5] Saving config report...")

    # Serialise bnb_config (may be a BitsAndBytesConfig object or dict)
    if isinstance(bnb_config, dict):
        config_data = bnb_config
    else:
        config_data = {
            "load_in_4bit"              : True,
            "bnb_4bit_quant_type"       : "nf4",
            "bnb_4bit_use_double_quant" : True,
            "bnb_4bit_compute_dtype"    : hw_info["recommended_dtype"],
        }

    report = {
        "week"           : 2,
        "task"           : "BitsAndBytes Configuration",
        "member"         : "R. Jeetesh Kumar",
        "hardware_used"  : hw_info,
        "bnb_config"     : config_data,
        "memory_savings" : memory_savings,
        "how_to_use"     : (
            "Pass bnb_config into FastLanguageModel.from_pretrained() "
            "as the quantization_config parameter. "
            "This replaces the default load_in_4bit=True in algoverse.ipynb "
            "with a fully explicit and documented config."
        ),
    }

    with open("bnb_config_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ── Final summary ──
    print("\n" + "=" * 55)
    print("  WEEK 2 SUMMARY")
    print("=" * 55)
    print(f"  Task          : BitsAndBytes Configuration")
    print(f"  Quant type    : NF4  (4-bit NormalFloat)")
    print(f"  Double quant  : Enabled  (saves ~0.4 GB extra)")
    print(f"  Compute dtype : {hw_info['recommended_dtype']}  (from Week 1 benchmark)")
    fp32_gb  = memory_savings.get("fp32",  {}).get("vram_needed_GB", "?")
    int4_gb  = memory_savings.get("int4",  {}).get("vram_needed_GB", "?")
    print(f"  VRAM: fp32 → {fp32_gb} GB  |  after 4-bit → {int4_gb} GB")
    print(f"  Reduction     : ~{round((fp32_gb - int4_gb) / fp32_gb * 100)}% less VRAM  🎯")
    print("=" * 55)
    print("\n  Config saved  → bnb_config_report.json")
    print("  Share with team before Week 3 fine-tuning.\n")
    print("  ── How to use in algoverse.ipynb ──")
    print("""
  from transformers import BitsAndBytesConfig
  import torch

  bnb_config = BitsAndBytesConfig(
      load_in_4bit              = True,
      bnb_4bit_quant_type       = "nf4",
      bnb_4bit_use_double_quant = True,
      bnb_4bit_compute_dtype    = torch.float16,  # or bfloat16
  )

  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name          = "unsloth/Phi-3-mini-4k-instruct",
      quantization_config = bnb_config,   # ← plug in here
      device_map          = "auto",
  )
  """)

    return report


# ─────────────────────────────────────────────────────────────
# MAIN — wire all 5 sections together
# ─────────────────────────────────────────────────────────────
def run_bnb_config():
    hw_info        = load_hardware_report()
    compute_dtype  = resolve_torch_dtype(hw_info["recommended_dtype"])
    bnb_config     = build_bnb_config(compute_dtype)
    memory_savings = estimate_memory_savings(hw_info["vram_total_GB"])
    report         = save_and_summarise(hw_info, bnb_config, memory_savings)
    return report


if __name__ == "__main__":
    run_bnb_config()
