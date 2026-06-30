"""
logger.py
=========
Centralized logging configuration for the Cricket Analytics & Match
Prediction Platform.

Why we use `logging` instead of `print()`:
    `print()` statements are fine for quick scripts, but in a production-style
    pipeline they have real downsides: you can't control verbosity, you can't
    easily redirect output to a file for later debugging, and you can't tell
    at a glance whether a message is informational, a warning, or a real
    error. The built-in `logging` module solves all of this: it supports
    severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL), timestamps,
    and simultaneous output to both the console (for live feedback while a
    script runs) and a log file (for after-the-fact debugging, e.g. "why did
    last night's training run fail?").

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Starting data preprocessing...")
    logger.warning("Found 12 rows with missing venue, dropping them.")
    logger.error("Failed to load matches.csv", exc_info=True)
"""

import logging
import sys

from src.utils.config import LOG_FILE_PATH, LOGS_DIR

# Track which loggers we've already configured so repeated calls to
# get_logger() with the same name don't attach duplicate handlers (which
# would cause every log message to print multiple times).
_CONFIGURED_LOGGERS: set[str] = set()

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a configured logger that writes to both the
    console and a shared pipeline log file.

    Args:
        name: Name of the logger, conventionally the calling module's
            `__name__`. This shows up in every log line so you know exactly
            which file produced a given message.
        level: Minimum severity level to log. Defaults to logging.INFO,
            meaning DEBUG-level messages are suppressed by default.

    Returns:
        A configured `logging.Logger` instance ready to use.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Loaded 1095 matches from matches.csv")
    """
    logger = logging.getLogger(name)

    # Guard against re-adding handlers if get_logger() is called multiple
    # times for the same module name (e.g. if a script is re-imported).
    if name in _CONFIGURED_LOGGERS:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # --- Console handler: prints to stdout so you see progress live ---
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler: appends to a persistent log file for later review ---
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent messages from propagating to the root logger, which would
    # otherwise cause duplicate output if the root logger also has handlers.
    logger.propagate = False

    _CONFIGURED_LOGGERS.add(name)
    return logger
