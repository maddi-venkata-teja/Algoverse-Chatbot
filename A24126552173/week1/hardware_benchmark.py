# ============================================================
# WEEK 1 — HARDWARE BENCHMARK
# AI Lab | Algoverse DSA Chatbot Project
# Member : R. Jeetesh Kumar (A24...173)
# ============================================================

import platform
import psutil
import subprocess
import json
import time
import sys
from datetime import datetime

# ─────────────────────────────────────────────
# SECTION 1 : CPU INFORMATION
# Why? The CPU handles data loading, tokenization,
# and preprocessing before data reaches the GPU.
# ─────────────────────────────────────────────
def get_cpu_info():
    print("\n[1/5] Scanning CPU...")
    info = {
        "processor"       : platform.processor() or platform.machine(),
        "physical_cores"  : psutil.cpu_count(logical=False),
        "logical_cores"   : psutil.cpu_count(logical=True),
        "max_freq_GHz"    : round(psutil.cpu_freq().max / 1000, 2) if psutil.cpu_freq() else "N/A",
        "current_usage_%"  : psutil.cpu_percent(interval=1),
        "architecture"    : platform.machine(),
    }
    return info


# ─────────────────────────────────────────────
# SECTION 2 : RAM INFORMATION
# Why? Fine-tuning loads the full model into RAM
# before offloading to GPU. Phi-3-mini (4-bit)
# needs ~4–6 GB free RAM to load safely.
# ─────────────────────────────────────────────
def get_ram_info():
    print("[2/5] Scanning RAM...")
    vm = psutil.virtual_memory()
    info = {
        "total_GB"     : round(vm.total     / (1024**3), 2),
        "available_GB" : round(vm.available / (1024**3), 2),
        "used_GB"      : round(vm.used      / (1024**3), 2),
        "usage_%"      : vm.percent,
        "sufficient_for_model": vm.available / (1024**3) >= 4.0,
    }
    return info


# ─────────────────────────────────────────────
# SECTION 3 : GPU INFORMATION
# Why? This is the MOST critical section.
# Unsloth + LoRA fine-tuning runs on GPU only.
# We check: VRAM, CUDA version, bf16 support.
# ─────────────────────────────────────────────
def get_gpu_info():
    print("[3/5] Scanning GPU...")
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "reason": "No CUDA GPU detected"}

        gpu_name  = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_free  = (torch.cuda.get_device_properties(0).total_memory
                      - torch.cuda.memory_allocated(0)) / (1024**3)

        # bf16 vs fp16 decision — used directly in algoverse.ipynb
        bf16_supported = torch.cuda.is_bf16_supported()

        info = {
            "available"       : True,
            "name"            : gpu_name,
            "vram_total_GB"   : round(vram_total, 2),
            "vram_free_GB"    : round(vram_free, 2),
            "cuda_version"    : torch.version.cuda,
            "torch_version"   : torch.__version__,
            "bf16_supported"  : bf16_supported,
            "recommended_dtype": "bf16" if bf16_supported else "fp16",
            # Phi-3-mini in 4-bit needs ~4 GB VRAM minimum
            "sufficient_vram" : vram_total >= 4.0,
        }

    except ImportError:
        # PyTorch not installed — try nvidia-smi as fallback
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                info = {
                    "available"      : True,
                    "name"           : parts[0],
                    "vram_total_GB"  : round(int(parts[1]) / 1024, 2),
                    "vram_free_GB"   : round(int(parts[2]) / 1024, 2),
                    "driver_version" : parts[3],
                    "note"           : "PyTorch not installed; read via nvidia-smi",
                }
            else:
                info = {"available": False, "reason": "nvidia-smi failed"}
        except FileNotFoundError:
            info = {"available": False, "reason": "No GPU / nvidia-smi not found"}

    return info


# ─────────────────────────────────────────────
# SECTION 4 : DISK SPEED TEST
# Why? Training datasets (dsa.json) and model
# checkpoints are read from disk. Slow disk =
# GPU sits idle waiting for data (I/O bottleneck).
# ─────────────────────────────────────────────
def get_disk_info():
    print("[4/5] Scanning Disk...")
    disk = psutil.disk_usage("/")

    # Quick sequential write speed test (~50 MB)
    test_file = "/tmp/_benchmark_jeetesh.bin"
    chunk      = b"J" * (1024 * 1024)   # 1 MB chunk
    mb_to_write = 50
    try:
        start = time.time()
        with open(test_file, "wb") as f:
            for _ in range(mb_to_write):
                f.write(chunk)
            f.flush()
        elapsed     = time.time() - start
        write_speed = round(mb_to_write / elapsed, 1)
    except Exception:
        write_speed = "N/A"
    finally:
        import os
        if os.path.exists(test_file):
            os.remove(test_file)

    info = {
        "total_GB"      : round(disk.total / (1024**3), 1),
        "free_GB"       : round(disk.free  / (1024**3), 1),
        "used_%"        : disk.percent,
        "write_speed_MBps": write_speed,
    }
    return info


# ─────────────────────────────────────────────
# SECTION 5 : OVERALL VERDICT
# Why? Summarises everything into a go/no-go
# recommendation for the team before training.
# ─────────────────────────────────────────────
def get_verdict(cpu, ram, gpu, disk):
    issues   = []
    warnings = []

    # GPU checks
    if not gpu.get("available"):
        issues.append("No GPU found — Unsloth fine-tuning REQUIRES a CUDA GPU.")
    elif not gpu.get("sufficient_vram", True):
        issues.append(f"VRAM too low ({gpu['vram_total_GB']} GB). Phi-3-mini needs ≥4 GB.")

    # RAM checks
    if not ram.get("sufficient_for_model", True):
        issues.append(f"Low RAM ({ram['available_GB']} GB free). Need ≥4 GB free.")

    # Disk checks
    if isinstance(disk.get("write_speed_MBps"), float) and disk["write_speed_MBps"] < 50:
        warnings.append(f"Slow disk write ({disk['write_speed_MBps']} MB/s). May cause I/O bottleneck.")

    if disk.get("free_GB", 100) < 10:
        warnings.append(f"Low disk space ({disk['free_GB']} GB free). Model + dataset need ~5 GB.")

    if issues:
        status = "NOT READY ✗"
    elif warnings:
        status = "READY WITH WARNINGS ⚠"
    else:
        status = "READY ✓"

    return {
        "status"   : status,
        "issues"   : issues,
        "warnings" : warnings,
        "recommended_dtype": gpu.get("recommended_dtype", "fp16"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────
# MAIN — Run all sections and print report
# ─────────────────────────────────────────────
def run_benchmark():
    print("=" * 55)
    print("  HARDWARE BENCHMARK — Algoverse AI Lab")
    print("  R. Jeetesh Kumar | Week 1 Task")
    print("=" * 55)

    cpu     = get_cpu_info()
    ram     = get_ram_info()
    gpu     = get_gpu_info()
    disk    = get_disk_info()
    verdict = get_verdict(cpu, ram, gpu, disk)

    print("[5/5] Computing verdict...\n")

    report = {
        "cpu"    : cpu,
        "ram"    : ram,
        "gpu"    : gpu,
        "disk"   : disk,
        "verdict": verdict,
    }

    # ── Pretty Print ──
    print("=" * 55)
    print("  BENCHMARK REPORT")
    print("=" * 55)

    sections = {
        "CPU"    : cpu,
        "RAM"    : ram,
        "GPU"    : gpu,
        "DISK"   : disk,
        "VERDICT": verdict,
    }

    for section, data in sections.items():
        print(f"\n  ── {section} ──")
        for key, val in data.items():
            label = key.replace("_", " ").ljust(22)
            print(f"    {label}: {val}")

    print("\n" + "=" * 55)
    print(f"  STATUS : {verdict['status']}")
    if verdict["issues"]:
        for i in verdict["issues"]:
            print(f"  ✗ {i}")
    if verdict["warnings"]:
        for w in verdict["warnings"]:
            print(f"  ⚠ {w}")
    print("=" * 55)

    # Save to JSON for team reference
    with open("hardware_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n  Report saved → hardware_report.json")
    print("  Share this file with your team before Week 3 training.\n")

    return report


if __name__ == "__main__":
    run_benchmark()
