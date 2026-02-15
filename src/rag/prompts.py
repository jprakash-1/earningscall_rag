"""Prompt templates for grounded answer synthesis with citations."""

from __future__ import annotations

SYSTEM_PROMPT = """You are an earnings-call research assistant.

Rules:
1. Use only the provided sources.
2. If evidence is missing, say so clearly.
3. Cite evidence using source labels like [S1], [S2].
4. Return STRICT JSON only with keys: answer, citation_ids.
5. citation_ids must be an array of strings that reference provided source labels.
"""

USER_PROMPT_TEMPLATE = """Question:
{query}

Sources:
{context}

Return strict JSON now."""
