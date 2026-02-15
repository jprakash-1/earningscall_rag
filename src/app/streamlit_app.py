"""Streamlit chat UI for the earnings-call agentic RAG assistant.

UI features:
- Chat conversation
- Optional metadata filters (company, section)
- Debug mode (router decision, retrieval snippets, latency)
- Expandable source citations
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from src.config import settings
from src.graph.graph import run_agentic_query
from src.utils.logging import configure_logging
from src.utils.tracing import configure_langsmith_tracing

configure_logging(debug=settings.debug)
configure_langsmith_tracing()

st.set_page_config(page_title="Earnings Call RAG", page_icon="📈", layout="wide")
st.title("Earnings Call Agentic RAG")
st.caption("Groq-powered generation + Pinecone retrieval with citations")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Controls")
    namespace = st.text_input("Pinecone namespace", value=settings.pinecone_namespace)
    company_filter = st.text_input("Company filter", value="")
    section_filter = st.selectbox("Section filter", options=["", "qa", "discussion", "transcript"], index=0)
    use_llm_router = st.checkbox("Use Groq router", value=True)
    debug_mode = st.checkbox("DEBUG mode", value=False)
    show_sources = st.checkbox("Show sources by default", value=True)

for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message["content"])

        citations = message.get("citations", [])
        if role == "assistant" and citations and show_sources:
            with st.expander("Sources", expanded=False):
                for citation in citations:
                    st.markdown(
                        f"**[{citation.get('citation_id', 'S?')}]** "
                        f"{citation.get('company', 'unknown')} | "
                        f"{citation.get('source', 'unknown')} | "
                        f"{citation.get('section', 'transcript')}"
                    )
                    st.caption(citation.get("snippet", ""))

        debug_payload = message.get("debug")
        if role == "assistant" and debug_mode and isinstance(debug_payload, dict):
            with st.expander("Debug details", expanded=False):
                st.json(debug_payload)

user_prompt = st.chat_input("Ask about earnings call transcripts...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    filters: dict[str, str] = {}
    if company_filter.strip():
        filters["company"] = company_filter.strip()
    if section_filter.strip():
        filters["section"] = section_filter.strip()

    with st.chat_message("assistant"):
        with st.spinner("Thinking and retrieving evidence..."):
            started = time.perf_counter()
            try:
                result = run_agentic_query(
                    user_prompt,
                    debug=debug_mode,
                    use_llm_router=use_llm_router,
                    namespace=namespace,
                    user_filters=filters,
                )
            except Exception as exc:
                result = {
                    "answer": (
                        "I could not complete the request. "
                        "Check GROQ/PINECONE/LANGSMITH env vars and dependencies."
                    ),
                    "citations": [],
                    "route": "error",
                    "error": str(exc),
                    "retrieved_chunks": [],
                }
            latency_ms = int((time.perf_counter() - started) * 1000)

        answer_text = str(result.get("answer", "No answer produced."))
        citations = result.get("citations", [])
        st.markdown(answer_text)

        if citations and show_sources:
            with st.expander("Sources", expanded=False):
                for citation in citations:
                    st.markdown(
                        f"**[{citation.get('citation_id', 'S?')}]** "
                        f"{citation.get('company', 'unknown')} | "
                        f"{citation.get('source', 'unknown')} | "
                        f"{citation.get('section', 'transcript')}"
                    )
                    st.caption(citation.get("snippet", ""))

        debug_payload: dict[str, Any] = {
            "route": result.get("route"),
            "route_reason": result.get("route_reason"),
            "filters": result.get("filters", {}),
            "latency_ms": latency_ms,
            "error": result.get("error"),
            "retrieved_snippets": [
                {
                    "citation_id": chunk.get("citation_id"),
                    "score": chunk.get("score"),
                    "text": str(chunk.get("text", ""))[:240],
                }
                for chunk in (result.get("retrieved_chunks") or [])[:3]
                if isinstance(chunk, dict)
            ],
        }
        if debug_mode:
            with st.expander("Debug details", expanded=False):
                st.json(debug_payload)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "citations": citations,
            "debug": debug_payload,
        }
    )
