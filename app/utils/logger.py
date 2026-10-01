"""
Central logging configuration for the Autonomous ML Platform.

Usage:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Stage started")
    logger.warning("Evidence coercion applied: %s", field)
    logger.debug("Raw LLM response: %s", response[:500])
    logger.exception("Pydantic validation failed")

Log file:   logs/app.log  (RotatingFileHandler, 10 MB x 5 backups)
Console:    yes
Level:      LOG_LEVEL env-var (default INFO)
Format:     timestamp | level | module:function:line | message

SECURITY: never log API keys, secret env-vars, or raw dataset rows.
          Log shapes, column names, and counts only.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "app.log"
_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
_BACKUP_COUNT = 5
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# ---------------------------------------------------------------------------
# Module-level sentinel so handlers are only added once
# ---------------------------------------------------------------------------
_configured: bool = False


def _configure_root() -> None:
    """Idempotently configure the root logger with file + console handlers."""
    global _configured
    if _configured:
        return

    # Create logs/ directory at runtime (never commit it)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    root = logging.getLogger()
    # Avoid duplicate handlers if module is reloaded in tests
    if not root.handlers:
        root.addHandler(file_handler)
        root.addHandler(console_handler)
    root.setLevel(level)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, initialising root handlers on first call."""
    _configure_root()
    return logging.getLogger(name)
