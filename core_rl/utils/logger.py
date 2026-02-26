# Beginner summary: This file provides a reusable logger setup so training scripts print clean, consistent logs.
from __future__ import annotations

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger for console output.

    Notes:
    - Reuses the same logger instance for a given name.
    - Adds a stream handler only once (prevents duplicate log lines).
    - Default level is INFO; callers can override (e.g., DEBUG in main.py).
    """
    logger = logging.getLogger(name)
    # Attach handler only if this logger has none yet.
    # Without this guard, repeated get_logger() calls would add duplicate handlers.
    if not logger.handlers:
        handler = logging.StreamHandler()
        # Example output:
        # 2026-02-23 23:54:11,432 | INFO | Starting training...
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Set the specified log level (default INFO, can be overridden to DEBUG, etc.)
    logger.setLevel(level)
    return logger
