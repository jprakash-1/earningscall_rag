"""State schema for the agentic RAG graph."""

from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    """Shared mutable state passed between LangGraph nodes."""

    query: str
    route: str
    route_reason: str
    filters: dict[str, Any]
    user_filters: dict[str, Any]
    namespace: str
    clarifying_questions: list[str]
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    debug: bool
    use_llm_router: bool
    error: str
