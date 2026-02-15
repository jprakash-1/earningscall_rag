"""LangGraph node implementations for agentic routing.

Routes:
- `retrieve`: use vector retrieval and grounded synthesis
- `clarify`: ask clarifying questions before retrieval
- `direct`: provide a direct conceptual answer without retrieval
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.graph.state import GraphState
from src.rag.chains import answer_query, synthesize_from_chunks
from src.rag.retriever import PineconeRetriever
from src.utils.llm import get_groq_chat_model
from src.utils.logging import get_logger
from src.utils.tracing import traceable

logger = get_logger(__name__)


AMBIGUOUS_HINTS = ("this", "that", "it", "they", "those", "these", "more about")
DIRECT_HINTS = ("what is", "explain", "how does", "difference between", "define")
RETRIEVE_SPECIFIC_HINTS = (
    "guidance",
    "quarter",
    "q1",
    "q2",
    "q3",
    "q4",
    "earnings call",
    "transcript",
    "tesla",
    "apple",
    "microsoft",
    "meta",
    "amazon",
)


def heuristic_route(query: str) -> tuple[str, str]:
    """Deterministic fallback router when LLM routing is unavailable.

    Why heuristics are useful:
    - Unit tests remain deterministic.
    - The app still behaves sensibly without external API credentials.
    """

    q = query.strip().lower()
    def has_ambiguous_hint() -> bool:
        for hint in AMBIGUOUS_HINTS:
            if " " in hint:
                if hint in q:
                    return True
            else:
                if re.search(rf"\b{re.escape(hint)}\b", q):
                    return True
        return False

    if any(q.startswith(prefix) for prefix in DIRECT_HINTS) and not any(
        hint in q for hint in RETRIEVE_SPECIFIC_HINTS
    ):
        return "direct", "Query asks for general explanation, not transcript evidence."

    if len(q) < 12 or has_ambiguous_hint():
        return "clarify", "Query is short or referential; clarification improves precision."

    return "retrieve", "Query appears specific enough for evidence-grounded retrieval."


@traceable(name="router_llm_decision", run_type="llm")
def _llm_route(query: str) -> tuple[str, str, dict[str, Any]]:
    """Use Groq to classify route and propose optional metadata filters."""

    llm = get_groq_chat_model()

    system = (
        "Classify user query for an earnings-call assistant. "
        "Return strict JSON with keys route, reason, filters. "
        "route must be one of retrieve, clarify, direct."
    )
    user = f"Query: {query}"

    raw = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )

    content = getattr(raw, "content", str(raw))
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = text.rstrip("`").strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Router LLM returned non-JSON output: {text}") from exc

    route = str(payload.get("route", "retrieve")).lower()
    if route not in {"retrieve", "clarify", "direct"}:
        route = "retrieve"

    reason = str(payload.get("reason", "No reason provided."))
    filters = payload.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}

    return route, reason, filters


@traceable(name="router_node", run_type="chain")
def router_node(state: GraphState) -> GraphState:
    """Choose routing path for current query."""

    query = state.get("query", "")
    use_llm_router = bool(state.get("use_llm_router", True))

    route: str
    reason: str
    filters: dict[str, Any] = {}

    if use_llm_router:
        try:
            route, reason, filters = _llm_route(query)
        except Exception as exc:
            route, reason = heuristic_route(query)
            filters = {}
            logger.warning(
                "LLM router failed; falling back to heuristic route",
                extra={"context": {"error": str(exc), "fallback_route": route}},
            )
    else:
        route, reason = heuristic_route(query)

    user_filters = state.get("user_filters")
    if isinstance(user_filters, dict):
        filters = {**filters, **user_filters}

    logger.info(
        "Router decision",
        extra={"context": {"route": route, "reason": reason, "filters": filters}},
    )

    return {
        **state,
        "route": route,
        "route_reason": reason,
        "filters": filters,
    }


def clarify_node(state: GraphState) -> GraphState:
    """Generate one or two clarifying questions for ambiguous queries."""

    query = state.get("query", "")
    questions = [
        f"Can you specify the company and quarter for: '{query}'?",
        "Do you want risks, guidance, or financial-performance details?",
    ]

    logger.info("Clarify node triggered", extra={"context": {"questions": len(questions)}})

    return {
        **state,
        "clarifying_questions": questions,
        "answer": "I need a bit more detail before retrieving evidence.\n- " + "\n- ".join(questions),
        "citations": [],
    }


@traceable(name="retrieve_node", run_type="retriever")
def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve chunks from Pinecone based on router decision and filters."""

    query = state.get("query", "")
    filters = state.get("filters") if isinstance(state.get("filters"), dict) else None
    namespace = str(state.get("namespace", "earnings-call-rag"))

    try:
        retriever = PineconeRetriever(namespace=namespace, top_k=6, use_mmr=False)
        chunks = retriever.retrieve(query, filters=filters)
        serialized = [
            {
                "citation_id": chunk.citation_id,
                "text": chunk.text,
                "score": chunk.score,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        logger.info("Retrieve node completed", extra={"context": {"chunks": len(serialized)}})
        return {**state, "retrieved_chunks": serialized}
    except Exception as exc:
        logger.error("Retrieve node failed", extra={"context": {"error": str(exc)}})
        return {**state, "retrieved_chunks": [], "error": str(exc)}


@traceable(name="synthesize_node", run_type="chain")
def synthesize_node(state: GraphState) -> GraphState:
    """Synthesize final grounded answer from retrieved evidence."""

    query = state.get("query", "")
    namespace = str(state.get("namespace", "earnings-call-rag"))
    try:
        retrieved_chunks = state.get("retrieved_chunks")
        if isinstance(retrieved_chunks, list):
            result = synthesize_from_chunks(query, retrieved_chunks)
        else:
            result = answer_query(
                query,
                namespace=namespace,
                top_k=6,
                use_mmr=False,
                filters=state.get("filters") if isinstance(state.get("filters"), dict) else None,
            )
        return {
            **state,
            "answer": result.answer,
            "citations": [citation.model_dump() for citation in result.citations],
        }
    except Exception as exc:
        logger.error("Synthesize node failed", extra={"context": {"error": str(exc)}})
        return {
            **state,
            "answer": "I could not synthesize an evidence-grounded answer.",
            "citations": [],
            "error": str(exc),
        }


@traceable(name="direct_answer_node", run_type="chain")
def direct_answer_node(state: GraphState) -> GraphState:
    """Provide direct conceptual help without retrieval."""

    query = state.get("query", "")

    try:
        llm = get_groq_chat_model()
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a finance assistant. Provide a concise conceptual answer "
                        "without citing transcript sources."
                    ),
                },
                {"role": "user", "content": query},
            ]
        )
        content = getattr(response, "content", str(response))
        answer = str(content).strip()
    except Exception:
        answer = "This is a general question. I can answer conceptually, or retrieve transcript evidence if you specify a company and quarter."

    logger.info("Direct answer node completed")
    return {**state, "answer": answer, "citations": []}
