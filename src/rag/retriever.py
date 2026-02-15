"""Retriever over Pinecone for earnings-call chunks.

Teaching design:
- We use Pinecone for vector search.
- We keep retrieval output structured (text + metadata + score).
- Optional MMR-like post-filtering can improve diversity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.indexing.embedder import get_embedder
from src.indexing.pinecone_client import get_or_create_index
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """One retrieved chunk from Pinecone."""

    citation_id: str
    text: str
    score: float
    metadata: dict[str, Any]


def _as_matches(query_result: Any) -> list[Any]:
    if hasattr(query_result, "matches"):
        return list(query_result.matches)
    if isinstance(query_result, dict):
        return list(query_result.get("matches", []))
    return []


def _as_metadata(match: Any) -> dict[str, Any]:
    if hasattr(match, "metadata") and isinstance(match.metadata, dict):
        return match.metadata
    if isinstance(match, dict) and isinstance(match.get("metadata"), dict):
        return match["metadata"]
    return {}


def _as_score(match: Any) -> float:
    if hasattr(match, "score"):
        return float(match.score)
    if isinstance(match, dict):
        return float(match.get("score", 0.0))
    return 0.0


def _mmr_diversify(chunks: list[RetrievedChunk], keep_k: int) -> list[RetrievedChunk]:
    """Simple diversity heuristic based on unique source/doc patterns."""

    if len(chunks) <= keep_k:
        return chunks

    selected: list[RetrievedChunk] = []
    seen_pairs: set[tuple[str, str]] = set()

    for chunk in chunks:
        doc_id = str(chunk.metadata.get("doc_id", ""))
        section = str(chunk.metadata.get("section", ""))
        key = (doc_id, section)

        if key not in seen_pairs:
            selected.append(chunk)
            seen_pairs.add(key)
            if len(selected) >= keep_k:
                break

    if len(selected) < keep_k:
        for chunk in chunks:
            if chunk not in selected:
                selected.append(chunk)
                if len(selected) >= keep_k:
                    break

    return selected


class PineconeRetriever:
    """Pinecone-backed retriever with optional diversity re-ranking."""

    def __init__(self, *, namespace: str, top_k: int = 6, use_mmr: bool = False) -> None:
        self.namespace = namespace
        self.top_k = top_k
        self.use_mmr = use_mmr

        self.embedder, self.embedding_model_name = get_embedder(prefer_hf=True)
        probe = self.embedder.embed_query("dimension probe")
        self.index = get_or_create_index(expected_dimension=len(probe))

    def retrieve(self, query: str, *, filters: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        """Retrieve chunks for a natural-language question."""

        query_vector = self.embedder.embed_query(query)

        raw = self.index.query(
            vector=query_vector,
            top_k=self.top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=filters or None,
        )

        matches = _as_matches(raw)
        chunks: list[RetrievedChunk] = []

        for idx, match in enumerate(matches, start=1):
            metadata = _as_metadata(match)
            text = str(metadata.get("text", ""))
            citation_id = f"S{idx}"
            chunks.append(
                RetrievedChunk(
                    citation_id=citation_id,
                    text=text,
                    score=_as_score(match),
                    metadata=metadata,
                )
            )

        if self.use_mmr:
            chunks = _mmr_diversify(chunks, keep_k=self.top_k)

        logger.info(
            "Retriever returned chunks",
            extra={"context": {"query": query, "count": len(chunks), "namespace": self.namespace}},
        )
        return chunks
