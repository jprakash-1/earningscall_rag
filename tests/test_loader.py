"""Tests for HuggingFace loader canonicalization behavior."""

from __future__ import annotations

from typing import Any

from src.data import hf_loader


def test_loader_small_mode_returns_records(monkeypatch: Any) -> None:
    """Loader should return non-empty records in small mode with key fields."""

    fake_rows = [
        {
            "question": "What changed in guidance?",
            "answer": "Guidance was raised for margins.",
            "transcript": "CEO: We are raising margin guidance.",
            "ticker": "TSLA",
            "quarter": "Q2",
        },
        {
            "question": "Any demand concerns?",
            "answer": "Demand remained resilient.",
            "transcript": "Analyst: Demand? CFO: Resilient.",
            "ticker": "AAPL",
        },
    ]

    def _fake_load_hf_split(dataset_name: str, split: str) -> list[dict[str, Any]]:
        assert dataset_name == hf_loader.DATASET_NAME
        assert split == "train"
        return fake_rows

    monkeypatch.setattr(hf_loader, "_load_hf_split", _fake_load_hf_split)

    records = hf_loader.load_records(mode="small", limit=2)
    assert records
    assert len(records) == 2

    for record in records:
        assert set(record.keys()) >= {"question", "answer", "text", "doc_id", "metadata"}
        assert record["question"]
        assert record["answer"]
        assert record["text"]
        assert record["doc_id"].startswith("doc_")
        assert isinstance(record["metadata"], dict)
