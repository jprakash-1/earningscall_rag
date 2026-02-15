"""Chunking strategies for earnings-call transcript documents.

This module provides two strategies:
1. Baseline: Recursive character chunking (LangChain when available).
2. Structure-aware: Heuristic splitting using speaker and Q&A markers.

Why two strategies:
- Baseline is robust and simple.
- Structure-aware chunks often preserve semantic boundaries in transcripts,
  which can improve retrieval relevance and citations.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from typing import Any

from src.data.doc_builder import records_to_documents
from src.data.hf_loader import load_records
from src.utils.ids import build_doc_id
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

SPEAKER_OR_QA_RE = re.compile(
    r"(?im)^\s*(operator|analyst|ceo|cfo|question|answer|q:|a:)\b"
)


def _chunk_text_manual(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Fallback chunk splitter that guarantees deterministic overlap behavior."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start = end - chunk_overlap

    return chunks


def _chunk_text_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Use LangChain recursive splitter when available; fallback otherwise."""

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)
    except ModuleNotFoundError:
        logger.warning(
            "langchain-text-splitters missing; using manual baseline splitter",
            extra={"context": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}},
        )
        return _chunk_text_manual(text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_baseline(
    documents: Iterable[Any],
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Any]:
    """Split documents with recursive character strategy."""

    docs = list(documents)
    chunked_docs: list[Any] = []

    for doc in docs:
        chunks = _chunk_text_recursive(doc.page_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, chunk_text in enumerate(chunks):
            metadata = dict(doc.metadata)
            chunk_id = build_doc_id(
                metadata.get("doc_id", "unknown"),
                "baseline",
                idx,
                chunk_text,
                prefix="chunk",
            )
            metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "split": "baseline",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
            )
            chunked_docs.append(type(doc)(page_content=chunk_text, metadata=metadata))

    logger.info(
        "Baseline chunking completed",
        extra={"context": {"input_docs": len(docs), "output_chunks": len(chunked_docs)}},
    )
    return chunked_docs


def _structure_units(text: str) -> list[str]:
    """Split transcript into structure-friendly units.

    Heuristics:
    - Primary split on blank lines.
    - Preserve speaker/Q&A prefixed lines as separate units.
    """

    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    units: list[str] = []

    for block in raw_blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if any(SPEAKER_OR_QA_RE.match(line) for line in lines):
            units.extend(lines)
        else:
            units.append(" ".join(lines))

    return units


def split_structure_aware(
    documents: Iterable[Any],
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[Any]:
    """Split documents using simple speaker/Q&A aware heuristics."""

    docs = list(documents)
    chunked_docs: list[Any] = []

    for doc in docs:
        units = _structure_units(doc.page_content)

        running: list[str] = []
        current_len = 0

        def flush_chunk(index: int) -> None:
            nonlocal running, current_len
            if not running:
                return
            chunk_text = "\n".join(running).strip()
            metadata = dict(doc.metadata)
            section = "qa" if SPEAKER_OR_QA_RE.search(chunk_text) else "discussion"
            chunk_id = build_doc_id(
                metadata.get("doc_id", "unknown"),
                "structure_aware",
                index,
                chunk_text,
                prefix="chunk",
            )
            metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "split": "structure_aware",
                    "section": section,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
            )
            chunked_docs.append(type(doc)(page_content=chunk_text, metadata=metadata))

            # Preserve char-overlap tail in a transparent way.
            if chunk_overlap > 0:
                tail = chunk_text[-chunk_overlap:]
                running = [tail]
                current_len = len(tail)
            else:
                running = []
                current_len = 0

        chunk_index = 0
        for unit in units:
            unit_len = len(unit)
            if current_len and current_len + 1 + unit_len > chunk_size:
                flush_chunk(chunk_index)
                chunk_index += 1

            running.append(unit)
            current_len += unit_len + (1 if current_len else 0)

        flush_chunk(chunk_index)

    logger.info(
        "Structure-aware chunking completed",
        extra={"context": {"output_chunks": len(chunked_docs)}},
    )
    return chunked_docs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk earnings-call documents using baseline and structure-aware strategies")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--print-samples", type=int, default=0)
    parser.add_argument("--debug", type=int, default=0)
    return parser


def main() -> None:
    """CLI entrypoint for chunking smoke runs."""

    args = _build_arg_parser().parse_args()
    configure_logging(debug=bool(args.debug))

    records = load_records(mode=args.mode, limit=args.limit)
    docs = records_to_documents(records)

    baseline_chunks = split_baseline(docs)
    structure_chunks = split_structure_aware(docs)

    summary = {
        "records": len(records),
        "documents": len(docs),
        "baseline_chunks": len(baseline_chunks),
        "structure_aware_chunks": len(structure_chunks),
        "baseline_samples": [
            {
                "text": doc.page_content[:200],
                "metadata": doc.metadata,
            }
            for doc in baseline_chunks[: args.print_samples]
        ],
        "structure_samples": [
            {
                "text": doc.page_content[:200],
                "metadata": doc.metadata,
            }
            for doc in structure_chunks[: args.print_samples]
        ],
    }

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
