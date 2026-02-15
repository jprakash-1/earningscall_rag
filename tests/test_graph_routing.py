"""Routing tests for the agentic decision layer."""

from __future__ import annotations

from src.graph.nodes import heuristic_route


def test_ambiguous_query_routes_to_clarify() -> None:
    """Short referential query should trigger clarify route."""

    route, _reason = heuristic_route("Can you explain this?")
    assert route == "clarify"


def test_specific_query_routes_to_retrieve() -> None:
    """Company/metric-specific query should trigger retrieval."""

    route, _reason = heuristic_route("Summarize guidance changes for Tesla Q2 earnings call")
    assert route == "retrieve"


def test_general_query_routes_to_direct() -> None:
    """General concept question should trigger direct answer route."""

    route, _reason = heuristic_route("What is operating margin and why does it matter?")
    assert route == "direct"
