"""
nexusvoice/src/logger.py
─────────────────────────────────────────────────────────────
Centralized logging setup.
Logs go to both console (via Rich) and a rotating file.
"""

import os
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(name: str = "nexusvoice") -> logging.Logger:
    """
    Configure root logger with file + console handlers.

    Returns:
        Configured logger instance.
    """
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    log_dir = Path(os.getenv("LOG_DIR", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "nexusvoice.log"

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.handlers:
        return logger  # Already configured

    # File handler — rotating, max 5MB, keeps last 3 files
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
