"""Logging utilities with optional structured output.

We prefer a central logger setup so every module emits logs in the same format.
Consistent logs are essential when debugging multi-step agentic workflows.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.config import settings


class JsonFormatter(logging.Formatter):
    """Convert standard Python log records into JSON strings.

    Why JSON logs:
    - Easy to parse in tools.
    - Easy to grep by key.
    - Stable shape for machine-readable diagnostics.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # If a module passes extra context as `record.context`, include it.
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload["context"] = context

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True)


def configure_logging(debug: bool | None = None) -> None:
    """Configure root logging once for the entire application.

    Parameters
    ----------
    debug:
        Optional explicit override. If `None`, use `settings.debug`.

    Teaching note:
    We clear existing handlers to avoid duplicate logs when modules are reloaded
    (common in notebooks and Streamlit).
    """

    effective_debug = settings.debug if debug is None else debug
    level = logging.DEBUG if effective_debug else logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers so repeated setup calls stay idempotent.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger.

    A small wrapper keeps call-sites clean and makes future enhancements easy.
    """

    return logging.getLogger(name)
