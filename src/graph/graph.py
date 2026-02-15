"""Graph construction and CLI for agentic routing."""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.config import settings
from src.graph.nodes import (
    clarify_node,
    direct_answer_node,
    retrieve_node,
    router_node,
    synthesize_node,
)
from src.graph.state import GraphState
from src.utils.logging import configure_logging, get_logger
from src.utils.tracing import configure_langsmith_tracing

logger = get_logger(__name__)


def _route_key(state: GraphState) -> str:
    """Read route value from state for conditional branching."""

    return str(state.get("route", "retrieve"))


def build_graph() -> Any:
    """Build and compile LangGraph state machine."""

    try:
        from langgraph.graph import END, START, StateGraph
    except ModuleNotFoundError as exc:
        raise RuntimeError("langgraph is not installed. Add dependency `langgraph`.") from exc

    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("direct", direct_answer_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_key,
        {
            "clarify": "clarify",
            "retrieve": "retrieve",
            "direct": "direct",
        },
    )

    graph.add_edge("clarify", END)
    graph.add_edge("direct", END)
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_agentic_query(
    query: str,
    *,
    debug: bool = False,
    use_llm_router: bool = True,
    namespace: str = settings.pinecone_namespace,
    user_filters: dict[str, Any] | None = None,
) -> GraphState:
    """Run full graph for a single query and return final state."""

    app = build_graph()
    initial_state: GraphState = {
        "query": query,
        "debug": debug,
        "use_llm_router": use_llm_router,
        "namespace": namespace,
        "user_filters": user_filters or {},
    }

    result = app.invoke(initial_state)
    logger.info(
        "Graph run completed",
        extra={"context": {"route": result.get("route"), "has_answer": bool(result.get("answer"))}},
    )
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agentic LangGraph routing query")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--use-llm-router", type=int, default=1)
    parser.add_argument("--namespace", type=str, default=settings.pinecone_namespace)
    parser.add_argument("--company", type=str, default="")
    parser.add_argument("--section", type=str, default="")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(debug=bool(args.debug))
    configure_langsmith_tracing()

    result = run_agentic_query(
        args.query,
        debug=bool(args.debug),
        use_llm_router=bool(args.use_llm_router),
        namespace=args.namespace,
        user_filters={k: v for k, v in {"company": args.company, "section": args.section}.items() if v},
    )

    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
