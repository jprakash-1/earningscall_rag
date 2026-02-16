"""End-to-end data pipeline runner for local inspection.

Stages covered:
1. Loader: fetch canonical records from HuggingFace (`hf_loader`).
2. Cleaner: apply text normalization preview (`cleaners`).
3. Builder: convert records into document objects (`doc_builder`).
4. Chunking: run baseline and structure-aware chunkers (`chunking`).

Usage:
    python -m src.data.run --mode small --limit 5 --samples 2
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.data.chunking import split_baseline, split_structure_aware
from src.data.cleaners import normalize_text
from src.data.doc_builder import records_to_documents
from src.data.hf_loader import load_records
from src.utils.logging import configure_logging


def _sample_records(records: list[dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records[:sample_count]:
        samples.append(
            {
                "doc_id": record.get("doc_id"),
                "question": record.get("question", "")[:120],
                "answer": record.get("answer", "")[:120],
                "text": record.get("text", "")[:180],
            }
        )
    return samples


def _clean_records_preview(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    cleaned_records: list[dict[str, Any]] = []
    changed_text_count = 0

    for record in records:
        cleaned = dict(record)
        text_before = record.get("text", "")
        cleaned_text = normalize_text(text_before)
        if cleaned_text != text_before:
            changed_text_count += 1

        cleaned["question"] = normalize_text(record.get("question", ""))
        cleaned["answer"] = normalize_text(record.get("answer", ""))
        cleaned["text"] = cleaned_text
        cleaned_records.append(cleaned)

    return cleaned_records, changed_text_count


def _sample_documents(documents: list[Any], sample_count: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for doc in documents[:sample_count]:
        samples.append(
            {
                "page_content": doc.page_content[:200],
                "metadata": {
                    "doc_id": doc.metadata.get("doc_id"),
                    "source_split": doc.metadata.get("source_split"),
                    "section": doc.metadata.get("section"),
                },
            }
        )
    return samples


def _sample_chunks(chunks: list[Any], sample_count: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for chunk in chunks[:sample_count]:
        samples.append(
            {
                "chunk_text": chunk.page_content[:200],
                "chunk_metadata": {
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "chunk_index": chunk.metadata.get("chunk_index"),
                    "split": chunk.metadata.get("split"),
                    "section": chunk.metadata.get("section"),
                    "doc_id": chunk.metadata.get("doc_id"),
                },
            }
        )
    return samples


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and inspect the full data preparation pipeline")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--debug", type=int, default=0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(debug=bool(args.debug))

    records = load_records(mode=args.mode, limit=args.limit, split=args.split)
    print(f"Loaded {len(records)} records from HuggingFace dataset (mode={args.mode}, split={args.split})")
    print("records sample:", records[:1])
    cleaned_records, changed_text_count = _clean_records_preview(records)
    documents = records_to_documents(cleaned_records, source_split=args.split)

    baseline_chunks = split_baseline(documents)
    structure_chunks = split_structure_aware(documents)

    summary = {
        "stage_counts": {
            "loader_records": len(records),
            "cleaner_records": len(cleaned_records),
            "cleaner_text_changed": changed_text_count,
            "builder_documents": len(documents),
            "chunking_baseline": len(baseline_chunks),
            "chunking_structure_aware": len(structure_chunks),
        },
        "samples": {
            "loader": _sample_records(records, args.samples),
            "cleaner": _sample_records(cleaned_records, args.samples),
            "builder": _sample_documents(documents, args.samples),
            "chunking_baseline": _sample_chunks(baseline_chunks, args.samples),
            "chunking_structure_aware": _sample_chunks(structure_chunks, args.samples),
        },
    }

    # print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
