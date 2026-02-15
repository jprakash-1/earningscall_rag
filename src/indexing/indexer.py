"""Pinecone indexing logic for chunked transcript documents."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _build_vector_id(metadata: dict[str, Any]) -> str:
    """Build stable vector ID of the form `doc_id:chunk_id`."""

    doc_id = str(metadata.get("doc_id", "unknown-doc"))
    chunk_id = str(metadata.get("chunk_id", metadata.get("chunk_index", "0")))
    return f"{doc_id}:{chunk_id}"


def _build_metadata(metadata: dict[str, Any], *, text: str) -> dict[str, Any]:
    """Normalize metadata keys required by retrieval and citations."""

    company = metadata.get("company") or metadata.get("ticker") or "unknown"
    source = metadata.get("source") or metadata.get("source_dataset") or "unknown"

    return {
        "company": str(company),
        "source": str(source),
        "doc_id": str(metadata.get("doc_id", "unknown-doc")),
        "chunk_id": str(metadata.get("chunk_id", "unknown-chunk")),
        "split": str(metadata.get("split", "baseline")),
        "section": str(metadata.get("section", "transcript")),
        "source_split": str(metadata.get("source_split", "train")),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "text": text,
    }


def upsert_documents(
    index: Any,
    documents: list[Any],
    embedder: Any,
    *,
    namespace: str,
    batch_size: int = 100,
) -> dict[str, Any]:
    """Embed and upsert chunked documents into Pinecone."""

    if not documents:
        return {"upserted": 0, "batches": 0}

    total = 0
    batches = 0

    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start : start + batch_size]
        texts = [doc.page_content for doc in batch_docs]
        vectors = embedder.embed_documents(texts)

        payload = []
        for doc, vector in zip(batch_docs, vectors, strict=True):
            metadata = _build_metadata(doc.metadata, text=doc.page_content)
            vector_id = _build_vector_id(doc.metadata)
            payload.append((vector_id, vector, metadata))

        index.upsert(vectors=payload, namespace=namespace)
        batches += 1
        total += len(payload)

        logger.info(
            "Upserted Pinecone batch",
            extra={"context": {"batch": batches, "batch_size": len(payload), "namespace": namespace}},
        )

    return {"upserted": total, "batches": batches}
