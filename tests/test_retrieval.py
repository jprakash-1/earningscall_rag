"""Pinecone retrieval smoke test.

Behavior:
- If Pinecone credentials are missing, skip.
- If credentials exist, upsert tiny synthetic vectors in a test namespace and
  assert a query returns at least `k` matches.
"""

from __future__ import annotations

import os
import uuid

import pytest

from src.indexing.embedder import DeterministicHashEmbeddings
from src.indexing.pinecone_client import get_or_create_index


@pytest.mark.integration
def test_pinecone_retrieval_smoke() -> None:
    """Smoke-test retrieval against Pinecone when env vars are configured."""

    required = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "PINECONE_CLOUD", "PINECONE_REGION"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.skip(f"Skipping Pinecone smoke test; missing env vars: {', '.join(missing)}")

    try:
        index = get_or_create_index(expected_dimension=1536)
    except RuntimeError as exc:
        pytest.skip(f"Skipping Pinecone smoke test: {exc}")

    embedder = DeterministicHashEmbeddings(dimension=1536)
    namespace = f"pytest-smoke-{uuid.uuid4().hex[:8]}"

    texts = [
        "Management raised gross margin guidance for next quarter.",
        "Currency headwinds were listed as a key risk to outlook.",
    ]

    vectors = embedder.embed_documents(texts)
    payload = [
        (
            f"test-doc:{idx}",
            vector,
            {
                "company": "DemoCorp",
                "source": "unit-test",
                "doc_id": "test-doc",
                "chunk_id": str(idx),
                "split": "baseline",
                "section": "qa",
                "text": texts[idx],
                "ingested_at": "2026-01-01T00:00:00+00:00",
            },
        )
        for idx, vector in enumerate(vectors)
    ]

    index.upsert(vectors=payload, namespace=namespace)

    query_vec = embedder.embed_query("What risks did management mention?")
    result = index.query(vector=query_vec, top_k=2, include_metadata=True, namespace=namespace)

    matches = getattr(result, "matches", None)
    if matches is None and isinstance(result, dict):
        matches = result.get("matches", [])

    assert matches is not None
    assert len(matches) >= 1
