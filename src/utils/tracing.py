"""LangSmith tracing utilities.

This helper keeps tracing optional but easy to enable. If LangSmith SDK is not
installed, these helpers degrade gracefully to no-op behavior.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

from src.config import settings

F = TypeVar("F", bound=Callable[..., Any])


def configure_langsmith_tracing() -> bool:
    """Configure tracing env flags and return whether tracing is enabled."""

    enabled = bool(settings.langchain_tracing_v2) and bool(settings.langsmith_api_key)
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else "false"

    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key

    return enabled


def is_tracing_enabled() -> bool:
    """Check if tracing is effectively enabled and authenticated."""

    return configure_langsmith_tracing()


def traceable(*, name: str, run_type: str = "chain") -> Callable[[F], F]:
    """Return LangSmith trace decorator if available, else no-op decorator."""

    try:
        from langsmith import traceable as langsmith_traceable  # type: ignore

        return langsmith_traceable(name=name, run_type=run_type)
    except Exception:
        def _decorator(func: F) -> F:
            return func

        return _decorator
