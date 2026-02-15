"""Tests for baseline and structure-aware chunking pipelines."""

from __future__ import annotations

from src.data.chunking import split_baseline, split_structure_aware
from src.data.doc_builder import records_to_documents


def _sample_records() -> list[dict[str, object]]:
    long_text = (
        "Operator: Welcome everyone to the quarter update. "
        "CEO: Revenue grew strongly and margins improved. "
        "Analyst: What are the risks for next quarter? "
        "CFO: We see currency headwinds and supply variability. "
        * 10
    )

    return [
        {
            "question": "What risks were mentioned?",
            "answer": "Currency and supply variability.",
            "text": long_text,
            "doc_id": "doc_test_001",
            "metadata": {"company": "DemoCorp", "source": "unit-test"},
        }
    ]


def test_chunk_counts_and_metadata_preserved() -> None:
    """Chunkers should produce chunks and retain metadata fields."""

    docs = records_to_documents(_sample_records())
    baseline = split_baseline(docs, chunk_size=200, chunk_overlap=40)
    structure = split_structure_aware(docs, chunk_size=220, chunk_overlap=30)

    assert len(baseline) > 0
    assert len(structure) > 0

    for chunk in baseline + structure:
        assert chunk.metadata["doc_id"] == "doc_test_001"
        assert chunk.metadata["company"] == "DemoCorp"
        assert "chunk_id" in chunk.metadata
        assert "split" in chunk.metadata


def test_baseline_chunk_lengths_and_overlap() -> None:
    """Baseline chunks should respect bounds and include overlap continuity."""

    docs = records_to_documents(_sample_records())
    chunks = split_baseline(docs, chunk_size=180, chunk_overlap=30)

    assert len(chunks) >= 2
    assert all(len(c.page_content) <= 180 for c in chunks)

    # Verify overlap by checking adjacent suffix/prefix continuity.
    previous = chunks[0].page_content
    current = chunks[1].page_content
    overlap_size = 10
    assert previous[-overlap_size:] in current
