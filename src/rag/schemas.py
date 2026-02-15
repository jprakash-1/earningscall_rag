"""Structured schema models for RAG responses with citations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """One citation reference tied to a retrieved chunk."""

    citation_id: str = Field(description="Stable citation label, e.g., S1")
    company: str = Field(default="unknown")
    source: str = Field(default="unknown")
    section: str = Field(default="transcript")
    snippet: str = Field(default="")
    score: float = Field(default=0.0)


class AnswerWithCitations(BaseModel):
    """Final response object returned by the RAG chain."""

    answer: str
    citations: list[Citation]
