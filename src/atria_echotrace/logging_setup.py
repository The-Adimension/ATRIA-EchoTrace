"""Logging configuration.

The notebook communicated progress with bare ``print`` calls throughout. A long-lived
server needs levelled, timestamped records instead, so those prints become log records
at the equivalent level.
"""

from __future__ import annotations

import logging
import logging.config
from typing import Any

LOGGER_NAME = "atria_echotrace"


def build_config(level: str = "INFO") -> dict[str, Any]:
    """Return a ``logging.config.dictConfig`` mapping for the application."""
    level = level.upper()
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            LOGGER_NAME: {"handlers": ["console"], "level": level, "propagate": False},
            "uvicorn": {"handlers": ["console"], "level": level, "propagate": False},
            "uvicorn.error": {"handlers": ["console"], "level": level, "propagate": False},
            # Access logs are noisy for a single-user clinical workstation.
            "uvicorn.access": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        },
        "root": {"handlers": ["console"], "level": "WARNING"},
    }


def configure(level: str = "INFO") -> None:
    """Install the logging configuration. Idempotent."""
    logging.config.dictConfig(build_config(level))


def get_logger(suffix: str | None = None) -> logging.Logger:
    """Return the application logger, optionally a dotted child of it."""
    return logging.getLogger(f"{LOGGER_NAME}.{suffix}" if suffix else LOGGER_NAME)
