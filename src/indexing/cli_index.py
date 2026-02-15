"""CLI entrypoint to build Pinecone index from earnings-call records.

Usage:
`python -m src.indexing.cli_index --mode small --limit 200`
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.config import settings
from src.data.chunking import split_baseline, split_structure_aware
from src.data.doc_builder import records_to_documents
from src.data.hf_loader import load_records
from src.indexing.embedder import get_embedder
from src.indexing.indexer import upsert_documents
from src.indexing.pinecone_client import get_or_create_index
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def run_indexing(
    *,
    mode: str,
    limit: int | None,
    strategy: str,
    namespace: str,
) -> dict[str, Any]:
    """Execute end-to-end indexing pipeline and return summary stats."""

    records = load_records(mode=mode, limit=limit)
    docs = records_to_documents(records)

    if strategy == "baseline":
        chunked = split_baseline(docs)
    elif strategy == "structure_aware":
        chunked = split_structure_aware(docs)
    else:
        raise ValueError("strategy must be one of: baseline, structure_aware")

    embedder, model_name = get_embedder(prefer_openai=True)

    # Determine embedding dimension once from a tiny probe vector.
    probe_vector = embedder.embed_query("dimension probe")
    index = get_or_create_index(expected_dimension=len(probe_vector))

    upsert_stats = upsert_documents(
        index=index,
        documents=chunked,
        embedder=embedder,
        namespace=namespace,
    )

    result = {
        "mode": mode,
        "limit": limit,
        "strategy": strategy,
        "records": len(records),
        "documents": len(docs),
        "chunks": len(chunked),
        "embedding_model": model_name,
        "embedding_dimension": len(probe_vector),
        "namespace": namespace,
        "upserted": upsert_stats["upserted"],
        "batches": upsert_stats["batches"],
    }
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Pinecone index from earnings-call records")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strategy", choices=["baseline", "structure_aware"], default="baseline")
    parser.add_argument("--namespace", type=str, default=settings.pinecone_namespace)
    parser.add_argument("--debug", type=int, default=0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(debug=bool(args.debug))

    logger.info(
        "Starting indexing CLI",
        extra={"context": {"mode": args.mode, "limit": args.limit, "strategy": args.strategy}},
    )

    result = run_indexing(
        mode=args.mode,
        limit=args.limit,
        strategy=args.strategy,
        namespace=args.namespace,
    )

    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
