"""HuggingFace dataset loader for `lamini/earnings-calls-qa`.

Teaching workflow for this module:
1. Fetch rows from HuggingFace using `datasets.load_dataset`.
2. Normalize field names into a canonical schema used by downstream code.
3. Provide a small mode for rapid local iteration.
4. Offer a CLI for quick inspection and smoke validation.

Canonical output schema per record:
{
    "question": str,
    "answer": str,
    "text": str,
    "doc_id": str,
    "metadata": dict[str, Any]
}
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.data.cleaners import normalize_metadata_value, normalize_text
from src.utils.ids import build_doc_id
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

DATASET_NAME = "lamini/earnings-calls-qa"

QUESTION_KEYS = ("question", "query", "prompt")
ANSWER_KEYS = ("answer", "response", "ground_truth", "label")
TEXT_KEYS = (
    "transcript",
    "text",
    "context",
    "content",
    "document",
    "passage",
)


def _pick_first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Pick the first non-empty string from candidate keys."""

    for key in keys:
        if key in row:
            value = normalize_text(row.get(key))
            if value:
                return value
    return ""


def _canonicalize_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    """Map a raw dataset row into the canonical record schema."""

    question = _pick_first_present(row, QUESTION_KEYS)
    answer = _pick_first_present(row, ANSWER_KEYS)
    text = _pick_first_present(row, TEXT_KEYS)

    if not text:
        # If transcript-like content is missing, concatenate question+answer so
        # we still have a searchable payload in development scenarios.
        text = normalize_text(f"Q: {question}\nA: {answer}")

    metadata: dict[str, Any] = {
        "source_dataset": DATASET_NAME,
        "row_index": row_index,
    }

    reserved = set(QUESTION_KEYS) | set(ANSWER_KEYS) | set(TEXT_KEYS)
    for key, value in row.items():
        if key in reserved:
            continue
        metadata[key] = normalize_metadata_value(value)

    doc_id = build_doc_id(
        row.get("id", ""),
        row.get("ticker", ""),
        question,
        text,
        prefix="doc",
    )

    record = {
        "question": question,
        "answer": answer,
        "text": text,
        "doc_id": doc_id,
        "metadata": metadata,
    }
    return record


def _load_hf_split(dataset_name: str, split: str) -> Any:
    """Load a dataset split from HuggingFace.

    Kept as a standalone function so tests can monkeypatch it cleanly.
    """
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'datasets' package is not installed. Install project dependencies "
            "to load HuggingFace datasets."
        ) from exc

    logger.info("Loading dataset split", extra={"context": {"dataset": dataset_name, "split": split}})
    return load_dataset(dataset_name, split=split)


def load_records(
    *,
    mode: str = "small",
    limit: int | None = None,
    dataset_name: str = DATASET_NAME,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Load and canonicalize records from HuggingFace.

    Parameters
    ----------
    mode:
        `small` limits record count for quick iteration. `full` streams all rows
        unless an explicit `limit` is provided.
    limit:
        Optional upper bound for records. In `small` mode, defaults to 100.
    dataset_name:
        HF dataset path.
    split:
        Dataset split name.
    """

    if mode not in {"small", "full"}:
        raise ValueError("mode must be either 'small' or 'full'")

    effective_limit = limit
    if effective_limit is None and mode == "small":
        effective_limit = 100

    try:
        hf_rows = _load_hf_split(dataset_name=dataset_name, split=split)
    except Exception as exc:  # pragma: no cover - fallback path is environment-dependent.
        logger.warning(
            "Failed to load HuggingFace dataset; using a tiny fallback sample",
            extra={"context": {"error": str(exc), "dataset": dataset_name, "split": split}},
        )
        fallback = [
            {
                "question": "What was management's outlook?",
                "answer": "Management expected moderate growth next quarter.",
                "transcript": "Operator: We now discuss guidance. CFO: We expect moderate growth.",
                "ticker": "DEMO",
            }
        ]
        return [_canonicalize_row(row, idx) for idx, row in enumerate(fallback)]

    records: list[dict[str, Any]] = []

    for idx, row in enumerate(hf_rows):
        if effective_limit is not None and idx >= effective_limit:
            break

        canonical = _canonicalize_row(dict(row), idx)
        if not canonical["question"] or not canonical["answer"]:
            logger.debug(
                "Row missing question/answer fields after normalization",
                extra={"context": {"row_index": idx, "doc_id": canonical["doc_id"]}},
            )
        records.append(canonical)

    logger.info(
        "Loaded canonical records",
        extra={"context": {"count": len(records), "mode": mode, "limit": effective_limit}},
    )
    return records


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load and preview earnings-call QA records from HuggingFace")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME)
    parser.add_argument("--debug", type=int, default=0)
    return parser


def main() -> None:
    """CLI entrypoint for loader smoke runs."""

    args = _build_arg_parser().parse_args()
    configure_logging(debug=bool(args.debug))

    records = load_records(
        mode=args.mode,
        limit=args.limit,
        split=args.split,
        dataset_name=args.dataset,
    )

    print(json.dumps({"count": len(records), "sample": records[:2]}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
