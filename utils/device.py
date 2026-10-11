"""
utils/device.py
---------------
GPU/CPU device selection with automatic CUDA detection.
Logs available hardware information at startup.
"""
import torch
from utils.logger import get_logger

logger = get_logger(__name__)


def get_device(preferred: str = "auto") -> torch.device:
    """
    Determine the best available compute device.

    Args:
        preferred: 'auto' | 'cuda' | 'mps' | 'cpu'

    Returns:
        torch.device instance.
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preferred)

    _log_device_info(device)
    return device


def _log_device_info(device: torch.device) -> None:
    """Log hardware information for the selected device."""
    logger.info(f"[bold green]Using device:[/bold green] {device}")

    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        logger.info(
            f"  GPU: {props.name}  |  "
            f"VRAM: {props.total_memory / 1e9:.1f} GB  |  "
            f"CUDA: {torch.version.cuda}  |  "
            f"PyTorch: {torch.__version__}"
        )
        logger.info(f"  cuDNN: {torch.backends.cudnn.version()}")
    elif device.type == "mps":
        logger.info("  Apple Silicon MPS backend enabled.")
    else:
        logger.info(
            f"  CPU fallback  |  PyTorch: {torch.__version__}  |  "
            f"Threads: {torch.get_num_threads()}"
        )


def log_memory_usage() -> dict:
    """
    Return current GPU memory usage as a dict.

    Returns:
        Dict with 'allocated_gb' and 'reserved_gb' keys (0 if no GPU).
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return {"allocated_gb": round(allocated, 3), "reserved_gb": round(reserved, 3)}
    return {"allocated_gb": 0.0, "reserved_gb": 0.0}
