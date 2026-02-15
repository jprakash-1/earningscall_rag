"""LangChain RAG chain that returns answers with citations."""

from __future__ import annotations

import argparse
import json
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.rag.retriever import PineconeRetriever
from src.rag.schemas import AnswerWithCitations, Citation
from src.utils.llm import get_groq_chat_model
from src.utils.logging import configure_logging, get_logger
from src.utils.tracing import configure_langsmith_tracing, traceable

logger = get_logger(__name__)


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Build prompt context with source labels for grounded synthesis."""

    lines: list[str] = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        company = metadata.get("company", "unknown")
        source = metadata.get("source", "unknown")
        section = metadata.get("section", "transcript")
        lines.append(
            f"[{chunk['citation_id']}] company={company} source={source} section={section}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(lines)


def _safe_parse_json(raw_text: str) -> dict[str, Any]:
    """Parse strict JSON, with resilient fallback for model wrappers."""

    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If parsing fails, we return a conservative fallback payload.
        return {"answer": raw_text, "citation_ids": []}


@traceable(name="synthesize_from_chunks", run_type="chain")
def synthesize_from_chunks(query: str, chunks: list[dict[str, Any]]) -> AnswerWithCitations:
    """Synthesize an answer from pre-retrieved chunks using Groq."""

    if not chunks:
        return AnswerWithCitations(
            answer="I could not find relevant evidence in the indexed transcripts.",
            citations=[],
        )

    llm = get_groq_chat_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    raw_response = chain.invoke({"query": query, "context": _format_context(chunks)})
    parsed = _safe_parse_json(raw_response)

    requested_ids = parsed.get("citation_ids", []) if isinstance(parsed, dict) else []
    requested_ids = [str(item) for item in requested_ids if isinstance(item, str)]

    by_id = {chunk["citation_id"]: chunk for chunk in chunks}
    citations: list[Citation] = []

    for citation_id in requested_ids:
        chunk = by_id.get(citation_id)
        if not chunk:
            continue
        metadata = chunk.get("metadata", {})
        citations.append(
            Citation(
                citation_id=citation_id,
                company=str(metadata.get("company", "unknown")),
                source=str(metadata.get("source", "unknown")),
                section=str(metadata.get("section", "transcript")),
                snippet=str(chunk.get("text", ""))[:280],
                score=float(chunk.get("score", 0.0)),
            )
        )

    answer_text = parsed.get("answer") if isinstance(parsed, dict) else None
    if not isinstance(answer_text, str) or not answer_text.strip():
        answer_text = "I could not produce a structured answer. Please refine the query."

    response = AnswerWithCitations(answer=answer_text.strip(), citations=citations)
    logger.info(
        "RAG chain completed",
        extra={"context": {"query": query, "citations": len(citations), "top_k": top_k}},
    )
    return response


@traceable(name="rag_answer_query", run_type="chain")
def answer_query(
    query: str,
    *,
    namespace: str,
    top_k: int = 6,
    use_mmr: bool = False,
    filters: dict[str, Any] | None = None,
) -> AnswerWithCitations:
    """Run retrieve-then-synthesize with Groq and structured citations."""

    retriever = PineconeRetriever(namespace=namespace, top_k=top_k, use_mmr=use_mmr)
    retrieved = retriever.retrieve(query, filters=filters)
    chunks = [
        {
            "citation_id": item.citation_id,
            "text": item.text,
            "score": item.score,
            "metadata": item.metadata,
        }
        for item in retrieved
    ]
    return synthesize_from_chunks(query, chunks)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAG query with citations over Pinecone")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--mode", choices=["pinecone"], default="pinecone")
    parser.add_argument("--namespace", type=str, default=settings.pinecone_namespace)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--use-mmr", type=int, default=0)
    parser.add_argument("--debug", type=int, default=0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(debug=bool(args.debug))
    configure_langsmith_tracing()

    result = answer_query(
        args.query,
        namespace=args.namespace,
        top_k=args.top_k,
        use_mmr=bool(args.use_mmr),
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
