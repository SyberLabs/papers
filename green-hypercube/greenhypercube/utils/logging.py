"""Minimal, dependency-free logging setup."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "greenhypercube") -> logging.Logger:
    """Return a process-wide configured logger.

    Idempotent: repeated calls do not stack handlers.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger("greenhypercube")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("greenhypercube") else f"greenhypercube.{name}")
