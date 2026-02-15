"""LLM factory helpers.

This project standardizes on Groq for all generation and routing tasks.
By centralizing model initialization here, we ensure every module uses the same
provider, model settings, and fail-fast validation behavior.
"""

from __future__ import annotations

import os
from typing import Any

from src.config import settings, validate_env


def get_groq_chat_model() -> Any:
    """Return a Groq chat model configured from environment settings.

    Raises
    ------
    RuntimeError
        If required Groq env vars are missing or package is not installed.
    """

    validate_env(["GROQ_API_KEY", "GROQ_MODEL"])

    try:
        from langchain_groq import ChatGroq  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "langchain-groq is not installed. Add `langchain-groq` to dependencies."
        ) from exc

    # Ensure provider SDK can resolve credentials even when caller does not
    # export environment variables manually.
    os.environ["GROQ_API_KEY"] = settings.groq_api_key or ""

    return ChatGroq(
        model=settings.groq_model,
        temperature=settings.groq_temperature,
    )
