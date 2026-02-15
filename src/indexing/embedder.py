"""Embedding model selection for indexing and retrieval.

Teaching notes:
- Primary path: HuggingFace sentence-transformer embeddings (non-OpenAI).
- Local/dev fallback: deterministic hash embeddings (no model download/API cost).

Why this setup:
Groq is used for generation/classification in this project, while embeddings are
handled by a separate provider suitable for vector search.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingsProtocol(Protocol):
    """Minimal embeddings protocol used by the indexer and retriever."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents into vectors."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string into one vector."""


@dataclass
class DeterministicHashEmbeddings:
    """Local deterministic embeddings for offline/small-mode development.

    Important:
    This is not semantically strong like sentence-transformer embeddings. Its
    purpose is to keep the pipeline runnable when model dependencies are absent.
    """

    dimension: int = 1536

    def _embed_one(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []

        current = seed
        while len(values) < self.dimension:
            current = hashlib.sha256(current).digest()
            values.extend(((byte / 255.0) * 2.0 - 1.0) for byte in current)

        return values[: self.dimension]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def get_embedder(prefer_hf: bool = True) -> tuple[EmbeddingsProtocol, str]:
    """Return an embeddings implementation and a human-readable model label."""

    if prefer_hf:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

            logger.info(
                "Using HuggingFace embeddings",
                extra={"context": {"model": settings.hf_embedding_model}},
            )
            return (
                HuggingFaceEmbeddings(model_name=settings.hf_embedding_model),
                settings.hf_embedding_model,
            )
        except ModuleNotFoundError:
            logger.warning(
                "langchain-huggingface missing; falling back to deterministic local embeddings"
            )
        except Exception as exc:
            # Model downloads can fail in restricted environments. We prefer a
            # deterministic fallback so local development remains unblocked.
            logger.warning(
                "Could not initialize HuggingFace embeddings; using deterministic fallback",
                extra={"context": {"error": str(exc), "model": settings.hf_embedding_model}},
            )

    logger.warning(
        "Using deterministic hash embeddings fallback",
        extra={"context": {"dimension": settings.fallback_embedding_dim}},
    )
    return DeterministicHashEmbeddings(dimension=settings.fallback_embedding_dim), "deterministic-hash"
