"""
utils/seed.py
-------------
Reproducibility utilities – set all random seeds across Python,
NumPy, and PyTorch (including cuDNN determinism).
"""
import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cuDNN determinism (slight performance cost)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


if __name__ == "__main__":
    set_seed(42)
    print("Seed set successfully (seed=42). Reproducibility initialized.")

