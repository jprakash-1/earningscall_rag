"""Evaluator definitions for answer quality and grounding.

We provide:
- `answer_correctness`: lexical overlap vs reference answer.
- `groundedness`: checks whether citations are present and referenced.
- `retrieval_relevance`: checks retrieval scores attached to citations.

These evaluators are implemented using LangSmith evaluator interfaces so they
can run inside LangSmith experiments.
"""

from __future__ import annotations

import math
import re
from typing import Any

from langsmith.evaluation import StringEvaluator, run_evaluator


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _f1_overlap(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, pred_count in pred_counts.items():
        overlap += min(pred_count, ref_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def _grade_correctness(input_text: str, prediction: str, answer: str | None) -> dict[str, Any]:
    reference = answer or ""
    score = _f1_overlap(prediction, reference)
    return {
        "score": float(round(score, 4)),
        "comment": f"Token-overlap F1 between prediction and reference: {score:.4f}",
    }


def _groundedness_eval(run: Any, example: Any | None) -> dict[str, Any]:
    outputs = getattr(run, "outputs", {}) or {}
    answer = str(outputs.get("answer", ""))
    citations = outputs.get("citations", [])

    has_citations = isinstance(citations, list) and len(citations) > 0
    cites_in_text = "[S" in answer

    score = 1.0 if has_citations else 0.0
    if has_citations and not cites_in_text:
        score = 0.75

    return {
        "key": "groundedness",
        "score": score,
        "comment": "Checks whether response includes citation-backed grounding.",
    }


def _retrieval_relevance_eval(run: Any, example: Any | None) -> dict[str, Any]:
    outputs = getattr(run, "outputs", {}) or {}
    citations = outputs.get("citations", [])

    if not isinstance(citations, list) or not citations:
        return {
            "key": "retrieval_relevance",
            "score": 0.0,
            "comment": "No citations returned, so retrieval relevance is zero.",
        }

    scores: list[float] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        raw = citation.get("score", 0.0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.0

        # Convert cosine-ish [-1, 1] score to [0, 1].
        normalized = max(0.0, min(1.0, (value + 1.0) / 2.0))
        scores.append(normalized)

    if not scores:
        return {
            "key": "retrieval_relevance",
            "score": 0.0,
            "comment": "Citations were present but had no numeric retrieval scores.",
        }

    mean_score = math.fsum(scores) / len(scores)
    return {
        "key": "retrieval_relevance",
        "score": float(round(mean_score, 4)),
        "comment": "Average normalized retrieval score across cited chunks.",
    }


def build_langsmith_evaluators() -> list[Any]:
    """Return evaluator objects compatible with LangSmith `evaluate(...)`."""

    correctness = StringEvaluator(
        evaluation_name="answer_correctness",
        input_key="question",
        prediction_key="answer",
        answer_key="answer",
        grading_function=_grade_correctness,
    )

    groundedness = run_evaluator(_groundedness_eval)
    retrieval_relevance = run_evaluator(_retrieval_relevance_eval)

    return [correctness, groundedness, retrieval_relevance]


def local_metric_bundle(prediction: str, reference: str, citations: list[dict[str, Any]]) -> dict[str, float]:
    """Compute local versions of eval metrics when remote evaluation is unavailable."""

    correctness = _f1_overlap(prediction, reference)
    grounded = 1.0 if citations else 0.0

    rel_scores: list[float] = []
    for citation in citations:
        raw = citation.get("score", 0.0)
        try:
            rel_scores.append(max(0.0, min(1.0, (float(raw) + 1.0) / 2.0)))
        except (TypeError, ValueError):
            pass

    relevance = (sum(rel_scores) / len(rel_scores)) if rel_scores else 0.0

    return {
        "answer_correctness": round(correctness, 4),
        "groundedness": round(grounded, 4),
        "retrieval_relevance": round(relevance, 4),
    }
