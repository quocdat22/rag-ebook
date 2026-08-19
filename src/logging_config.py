"""Logging setup: one consistent format, level from env (LOG_LEVEL, default INFO)."""

import logging

from src.config import Settings

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

_configured = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once; level from `LOG_LEVEL` env (default INFO)."""
    global _configured
    if _configured:
        return
    logging.basicConfig(format=FORMAT, level=(level or Settings().log_level).upper())
    _configured = True
