"""
utils/logger.py
---------------
Centralized logging utility for the Deepfake Detection Platform.
Provides colored console output, file logging, and TensorBoard support.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


_loggers: dict[str, logging.Logger] = {}
_console = Console(stderr=True)


def get_logger(
    name: str,
    log_dir: Optional[str | Path] = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Get or create a named logger.

    Args:
        name:    Logger name (usually __name__).
        log_dir: Directory to write log files. If None, logs only to console.
        level:   Logging level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # ── Rich console handler ───────────────────────────────────
    rich_handler = RichHandler(
        console=_console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    # ── File handler (optional) ────────────────────────────────
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name.replace('.', '_')}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger
