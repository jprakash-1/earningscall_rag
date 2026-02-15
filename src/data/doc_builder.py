"""Convert canonical records into LangChain-style `Document` objects.

Teaching context:
Downstream RAG components expect a common document structure with two fields:
- `page_content`: the text to embed and retrieve
- `metadata`: structured attributes used for filtering and citations

This module creates that boundary cleanly so indexing/retrieval code does not
need to know the original dataset schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.cleaners import normalize_text


@dataclass
class SimpleDocument:
    """Fallback document type when LangChain is unavailable.

    This mirrors the minimal interface used throughout the project.
    """

    page_content: str
    metadata: dict[str, Any]


def _resolve_document_class() -> type:
    """Return LangChain `Document` if available, else fallback dataclass."""

    try:
        from langchain_core.documents import Document  # type: ignore

        return Document
    except ModuleNotFoundError:
        return SimpleDocument


def record_to_document(record: dict[str, Any], *, source_split: str = "train") -> Any:
    """Convert one canonical record into a document object.

    Parameters
    ----------
    record:
        Canonical record with keys `text`, `doc_id`, and `metadata`.
    source_split:
        Dataset split annotation for provenance.
    """

    DocumentClass = _resolve_document_class()

    base_metadata = dict(record.get("metadata", {}))
    base_metadata.update(
        {
            "doc_id": record.get("doc_id"),
            "source_split": source_split,
            "question": record.get("question", ""),
            "answer": record.get("answer", ""),
            "section": base_metadata.get("section", "transcript"),
        }
    )

    return DocumentClass(
        page_content=normalize_text(record.get("text", "")),
        metadata=base_metadata,
    )


def records_to_documents(records: list[dict[str, Any]], *, source_split: str = "train") -> list[Any]:
    """Batch-convert records into document objects."""

    return [record_to_document(record, source_split=source_split) for record in records]
