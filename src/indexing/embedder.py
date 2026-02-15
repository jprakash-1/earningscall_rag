"""Embedding model selection for indexing and retrieval.

Teaching notes:
- Production path: OpenAI embeddings through LangChain.
- Local/dev fallback: deterministic hash embeddings (no API cost, no network).

The fallback keeps development loops fast while still letting us exercise end-to-
end indexing logic and tests.
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
    This is not semantically strong like real embedding models. Its purpose is
    to keep the pipeline runnable when API keys are unavailable.
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


def get_embedder(prefer_openai: bool = True) -> tuple[EmbeddingsProtocol, str]:
    """Return an embeddings implementation and a human-readable model label."""

    if prefer_openai and settings.openai_api_key:
        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore

            logger.info(
                "Using OpenAI embeddings",
                extra={"context": {"model": settings.openai_embedding_model}},
            )
            return (
                OpenAIEmbeddings(
                    model=settings.openai_embedding_model,
                    api_key=settings.openai_api_key,
                ),
                settings.openai_embedding_model,
            )
        except ModuleNotFoundError:
            logger.warning(
                "langchain-openai not installed; falling back to deterministic local embeddings"
            )

    logger.warning(
        "Using deterministic hash embeddings fallback",
        extra={"context": {"dimension": settings.fallback_embedding_dim}},
    )
    return DeterministicHashEmbeddings(dimension=settings.fallback_embedding_dim), "deterministic-hash"
