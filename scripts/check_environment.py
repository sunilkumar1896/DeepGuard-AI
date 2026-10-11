"""
scripts/check_environment.py
----------------------------
Run this first to verify your environment is correctly set up.
Reports Python version, GPU availability, and all package versions.

Usage:
    python scripts/check_environment.py
"""
import sys
import platform
import subprocess
from pathlib import Path
from datetime import datetime

# ── Add project root to path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_python() -> bool:
    """Verify Python version >= 3.9."""
    v = sys.version_info
    ok = v >= (3, 9)
    status = "✅" if ok else "❌"
    print(f"{status} Python {v.major}.{v.minor}.{v.micro} "
          f"({'OK' if ok else 'Need >= 3.9'})")
    print(f"   Path: {sys.executable}")
    return ok


def check_package(import_name: str, display_name: str | None = None) -> bool:
    """Try importing a package and print its version."""
    name = display_name or import_name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  ✅ {name:<25} {version}")
        return True
    except ImportError as e:
        print(f"  ❌ {name:<25} NOT INSTALLED  ({e})")
        return False


def check_torch() -> bool:
    """Detailed PyTorch / CUDA check."""
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"  ✅ PyTorch               {torch.__version__}")
        print(f"     CUDA available:       {cuda_ok}")
        if cuda_ok:
            print(f"     CUDA version:         {torch.version.cuda}")
            n = torch.cuda.device_count()
            for i in range(n):
                p = torch.cuda.get_device_properties(i)
                gb = p.total_memory / 1024**3
                print(f"     GPU [{i}]:             {p.name}  ({gb:.1f} GB)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("     MPS (Apple Silicon):  available")
        else:
            print("     ⚠️  No GPU found — will run on CPU (training will be slow)")
        return True
    except ImportError:
        print("  ❌ PyTorch               NOT INSTALLED")
        print("     Install with one of:")
        print("       CUDA 12.1: pip install torch torchvision torchaudio "
              "--index-url https://download.pytorch.org/whl/cu121")
        print("       CPU only:  pip install torch torchvision torchaudio "
              "--index-url https://download.pytorch.org/whl/cpu")
        return False


def check_opencv() -> bool:
    """Check OpenCV build info."""
    try:
        import cv2
        print(f"  ✅ OpenCV                {cv2.__version__}")
        # Check if CUDA-enabled build
        info = cv2.getBuildInformation()
        cuda_in_cv = "CUDA" in info and "YES" in info[info.find("CUDA"):]
        if cuda_in_cv:
            print("     OpenCV CUDA:          YES")
        return True
    except ImportError:
        print("  ❌ OpenCV                NOT INSTALLED")
        return False


def check_config() -> bool:
    """Validate config loading."""
    try:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        print(f"  ✅ Config loaded         {config_path.name}")
        print(f"     Image size:           {cfg['data']['image_size']}")
        print(f"     CNN backbone:         {cfg['model']['cnn_backbone']}")
        return True
    except Exception as e:
        print(f"  ❌ Config failed:        {e}")
        return False


def check_disk_space() -> None:
    """Warn if disk space is below 20 GB."""
    import shutil
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free / 1024**3
    flag = "✅" if free_gb >= 20 else "⚠️ "
    print(f"  {flag} Disk free:           {free_gb:.1f} GB "
          f"({'OK' if free_gb >= 20 else 'Low — need ~20 GB for datasets'})")


def main() -> None:
    print("=" * 65)
    print(" Deepfake Detection Platform — Environment Check")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    print("\n[1] Python")
    py_ok = check_python()

    print("\n[2] Deep Learning")
    torch_ok = check_torch()

    print("\n[3] Computer Vision")
    cv_ok = check_opencv()
    check_package("PIL", "Pillow")
    check_package("albumentations")

    print("\n[4] Face Detection")
    check_package("facenet_pytorch", "FaceNet-PyTorch")
    check_package("mediapipe")
    check_package("insightface")

    print("\n[5] ML / Science")
    check_package("numpy")
    check_package("pandas")
    check_package("sklearn", "scikit-learn")
    check_package("scipy")
    check_package("matplotlib")
    check_package("seaborn")
    check_package("tqdm")
    check_package("timm")
    check_package("einops")

    print("\n[6] Explainability")
    check_package("pytorch_grad_cam", "grad-cam")

    print("\n[7] Training / Logging")
    check_package("tensorboard")
    check_package("wandb")

    print("\n[8] Backend")
    check_package("fastapi")
    check_package("uvicorn")
    check_package("sqlalchemy")
    check_package("pydantic")

    print("\n[9] Project Config")
    check_config()

    print("\n[10] System Resources")
    print(f"     OS:                   {platform.system()} {platform.release()}")
    import psutil
    ram_gb = psutil.virtual_memory().total / 1024**3
    print(f"  {'✅' if ram_gb >= 8 else '⚠️ '} RAM:                  "
          f"{ram_gb:.1f} GB {'(OK)' if ram_gb >= 8 else '(Low — need >= 8 GB)'}")
    check_disk_space()

    print("\n" + "=" * 65)
    if not torch_ok:
        print("⚠️  ACTION REQUIRED: Install PyTorch first!")
        print("   Run: pip install -r requirements.txt")
        print("   Then install PyTorch from: https://pytorch.org/get-started/locally/")
    else:
        print("✅ Environment looks good! Proceed to Stage 2: Dataset Preparation.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
