"""Run LangSmith evaluation experiments and write markdown summary report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from src.config import settings
from src.data.hf_loader import load_records
from src.eval.evaluators import build_langsmith_evaluators, local_metric_bundle
from src.indexing.cli_index import run_indexing
from src.rag.chains import answer_query
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "baseline": {
        "strategy": "baseline",
        "top_k": 6,
        "use_mmr": False,
        "namespace_suffix": "baseline",
    },
    "improved": {
        "strategy": "structure_aware",
        "top_k": 8,
        "use_mmr": True,
        "namespace_suffix": "improved",
    },
}


def _experiment_namespace(experiment: str) -> str:
    return f"{settings.pinecone_namespace}-{EXPERIMENTS[experiment]['namespace_suffix']}"


def _load_local_eval_examples(limit: int) -> list[dict[str, str]]:
    records = load_records(mode="small", limit=limit)
    examples: list[dict[str, str]] = []
    for record in records:
        question = str(record.get("question", "")).strip()
        answer = str(record.get("answer", "")).strip()
        if question and answer:
            examples.append({"question": question, "answer": answer})
    return examples


def _run_local_eval(experiment: str, limit: int) -> dict[str, Any]:
    cfg = EXPERIMENTS[experiment]
    namespace = _experiment_namespace(experiment)

    examples = _load_local_eval_examples(limit)
    metrics_accumulator: dict[str, list[float]] = {
        "answer_correctness": [],
        "groundedness": [],
        "retrieval_relevance": [],
    }

    failures = 0
    for example in examples:
        try:
            result = answer_query(
                example["question"],
                namespace=namespace,
                top_k=cfg["top_k"],
                use_mmr=cfg["use_mmr"],
            )
            metrics = local_metric_bundle(
                prediction=result.answer,
                reference=example["answer"],
                citations=[citation.model_dump() for citation in result.citations],
            )
            for key, value in metrics.items():
                metrics_accumulator[key].append(value)
        except Exception as exc:
            failures += 1
            logger.warning(
                "Local eval example failed",
                extra={"context": {"question": example["question"], "error": str(exc)}},
            )

    summary = {
        "mode": "local",
        "experiment": experiment,
        "namespace": namespace,
        "examples": len(examples),
        "failures": failures,
        "metrics": {
            key: (round(mean(values), 4) if values else 0.0)
            for key, values in metrics_accumulator.items()
        },
        "langsmith_run": None,
    }
    return summary


def _run_langsmith_eval(experiment: str, dataset_name: str) -> dict[str, Any]:
    cfg = EXPERIMENTS[experiment]
    namespace = _experiment_namespace(experiment)

    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client(api_key=settings.langsmith_api_key)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs.get("question", ""))
        result = answer_query(
            question,
            namespace=namespace,
            top_k=cfg["top_k"],
            use_mmr=cfg["use_mmr"],
        )
        return {
            "answer": result.answer,
            "citations": [citation.model_dump() for citation in result.citations],
        }

    evaluators = build_langsmith_evaluators()
    experiment_results = evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=f"earningscall-{experiment}",
        metadata={"experiment": experiment, **cfg, "namespace": namespace},
        max_concurrency=2,
        client=client,
        upload_results=True,
    )

    dataframe = experiment_results.to_pandas()
    numeric_columns = [
        col
        for col in dataframe.columns
        if any(name in col for name in ["answer_correctness", "groundedness", "retrieval_relevance"])
    ]

    metric_means: dict[str, float] = {}
    for column in numeric_columns:
        try:
            metric_means[column] = round(float(dataframe[column].mean()), 4)
        except Exception:
            continue

    return {
        "mode": "langsmith",
        "experiment": experiment,
        "namespace": namespace,
        "examples": int(len(dataframe.index)),
        "failures": int(dataframe["error"].notna().sum()) if "error" in dataframe.columns else 0,
        "metrics": metric_means,
        "langsmith_run": experiment_results.experiment_name,
    }


def _write_markdown_report(summary: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Evaluation Summary",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Mode: {summary['mode']}",
        f"- Experiment: {summary['experiment']}",
        f"- Namespace: {summary['namespace']}",
        f"- Examples: {summary['examples']}",
        f"- Failures: {summary['failures']}",
        "",
        "## Metrics",
    ]

    metrics = summary.get("metrics", {})
    if metrics:
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No metrics available")

    lines.extend(
        [
            "",
            "## LangSmith Run",
            f"- {summary.get('langsmith_run') or 'Not uploaded (local mode)'}",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LangSmith evaluation experiments")
    parser.add_argument("--experiment", choices=["baseline", "improved"], required=True)
    parser.add_argument("--dataset-name", type=str, default="earningscall-rag-eval")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--index-limit", type=int, default=200)
    parser.add_argument("--skip-index", type=int, default=0)
    parser.add_argument("--report", type=str, default="reports/eval_summary.md")
    parser.add_argument("--debug", type=int, default=0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(debug=bool(args.debug))

    cfg = EXPERIMENTS[args.experiment]
    namespace = _experiment_namespace(args.experiment)

    if not bool(args.skip_index):
        logger.info(
            "Running indexing before evaluation",
            extra={
                "context": {
                    "experiment": args.experiment,
                    "strategy": cfg["strategy"],
                    "namespace": namespace,
                    "index_limit": args.index_limit,
                }
            },
        )
        try:
            run_indexing(
                mode=args.mode,
                limit=args.index_limit,
                strategy=cfg["strategy"],
                namespace=namespace,
            )
        except Exception as exc:
            logger.warning(
                "Indexing step failed; continuing with available index/local mode",
                extra={"context": {"error": str(exc)}},
            )

    if settings.langsmith_api_key:
        try:
            summary = _run_langsmith_eval(args.experiment, args.dataset_name)
        except Exception as exc:
            logger.warning(
                "LangSmith eval failed; falling back to local mode",
                extra={"context": {"error": str(exc)}},
            )
            summary = _run_local_eval(args.experiment, limit=args.limit)
    else:
        logger.warning("LANGSMITH_API_KEY missing. Falling back to local eval mode.")
        summary = _run_local_eval(args.experiment, limit=args.limit)

    report_path = Path(args.report)
    _write_markdown_report(summary, report_path)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Markdown report: {report_path}")


if __name__ == "__main__":
    main()
