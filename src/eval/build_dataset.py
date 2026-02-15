"""Build a LangSmith evaluation dataset from HuggingFace QA records.

Primary flow:
1. Load canonical records from HF loader.
2. Map each record to LangSmith example format.
3. Upsert examples into a named LangSmith dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import settings, validate_env
from src.data.hf_loader import load_records
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _examples_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert canonical records into LangSmith example payloads."""

    examples: list[dict[str, Any]] = []
    for record in records:
        examples.append(
            {
                "inputs": {"question": record.get("question", "")},
                "outputs": {"answer": record.get("answer", "")},
                "metadata": {
                    "doc_id": record.get("doc_id"),
                    "source_dataset": record.get("metadata", {}).get("source_dataset"),
                },
            }
        )
    return examples


def build_langsmith_dataset(*, dataset_name: str, mode: str, limit: int) -> dict[str, Any]:
    """Create/update LangSmith dataset and upload examples."""

    validate_env(["LANGSMITH_API_KEY", "LANGSMITH_PROJECT"])

    from langsmith import Client

    client = Client(api_key=settings.langsmith_api_key)
    records = load_records(mode=mode, limit=limit)
    examples = _examples_from_records(records)

    existing = next(client.list_datasets(dataset_name=dataset_name, limit=1), None)
    if existing is None:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Earnings-call QA pairs derived from lamini/earnings-calls-qa",
            metadata={"source": "lamini/earnings-calls-qa"},
        )
    else:
        dataset = existing

    client.create_examples(dataset_id=dataset.id, examples=examples)

    return {
        "dataset_name": dataset_name,
        "dataset_id": str(dataset.id),
        "examples_uploaded": len(examples),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LangSmith dataset from earnings-call QA records")
    parser.add_argument("--dataset-name", type=str, default="earningscall-rag-eval")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument(
        "--local-dump",
        type=str,
        default="reports/eval_dataset_preview.json",
        help="Always save a local JSON preview of examples for reproducibility.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(debug=bool(args.debug))

    records = load_records(mode=args.mode, limit=args.limit)
    examples = _examples_from_records(records)

    local_dump_path = Path(args.local_dump)
    local_dump_path.parent.mkdir(parents=True, exist_ok=True)
    local_dump_path.write_text(json.dumps(examples, indent=2, ensure_ascii=True), encoding="utf-8")

    result: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "examples_prepared": len(examples),
        "local_dump": str(local_dump_path),
        "uploaded_to_langsmith": False,
    }

    if settings.langsmith_api_key and settings.langsmith_project:
        upload_result = build_langsmith_dataset(
            dataset_name=args.dataset_name,
            mode=args.mode,
            limit=args.limit,
        )
        result.update(upload_result)
        result["uploaded_to_langsmith"] = True
    else:
        logger.warning(
            "Skipping LangSmith upload because LANGSMITH_API_KEY/LANGSMITH_PROJECT is missing"
        )

    logger.info("LangSmith dataset prepared", extra={"context": result})
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
