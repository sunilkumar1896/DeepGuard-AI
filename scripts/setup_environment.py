# -*- coding: utf-8 -*-
"""
scripts/setup_environment.py
----------------------------
One-click environment setup script.
Detects GPU, installs correct PyTorch variant, then all other requirements.

Usage:
    python scripts/setup_environment.py [--cpu-only]
"""
import subprocess
import sys
import os
import argparse
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command and stream output."""
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=check)
    return result


def detect_cuda() -> str | None:
    """Try to detect CUDA version via nvcc or nvidia-smi."""
    # Try nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # If nvidia-smi works, assume CUDA 12.1+ (most common)
        return "cu121"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Try nvcc
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.DEVNULL, text=True
        )
        if "12." in out:
            return "cu121"
        elif "11.8" in out:
            return "cu118"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setup environment for Deepfake Detection Platform"
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Force CPU-only PyTorch installation"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" Deepfake Detection Platform - Environment Setup")
    print("=" * 60)

    pip = [sys.executable, "-m", "pip"]

    # 1. Upgrade pip
    run([*pip, "install", "--upgrade", "pip"])

    # 2. Install PyTorch
    if args.cpu_only:
        cuda_tag = None
    else:
        cuda_tag = detect_cuda()

    if cuda_tag:
        index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
        print(f"\n[OK] CUDA detected ({cuda_tag}) - installing GPU version of PyTorch")
        run([
            *pip, "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", index_url,
        ])
    else:
        print("\n[WARN] No CUDA detected - installing CPU-only PyTorch")
        print("   Training will work but may be SLOW without a GPU.")
        print("   If you have a GPU, install CUDA drivers and re-run this script.")
        run([
            *pip, "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ])

    # 3. Install remaining requirements
    req_file = Path(__file__).parent.parent / "requirements.txt"
    print(f"\n📦 Installing remaining packages from {req_file.name} ...")
    run([*pip, "install", "-r", str(req_file), "--no-deps", "--quiet"])
    # Install with deps for key packages
    run([*pip, "install", "-r", str(req_file)])

    print("\n" + "=" * 60)
    print("[DONE] Setup complete!")
    print("   Now run: python scripts/check_environment.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
