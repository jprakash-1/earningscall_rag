"""Pinecone initialization helpers.

Responsibilities:
- Validate required Pinecone environment variables.
- Create index if it does not exist.
- Validate index dimension compatibility with selected embeddings model.
"""

from __future__ import annotations

from typing import Any

from src.config import settings, validate_env
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_pinecone_sdk() -> tuple[Any, Any]:
    """Import Pinecone SDK lazily to keep module import lightweight."""

    try:
        from pinecone import Pinecone, ServerlessSpec  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pinecone SDK is not installed. Add dependency `pinecone` to run indexing."
        ) from exc
    return Pinecone, ServerlessSpec


def ensure_pinecone_env() -> None:
    """Fail fast when required Pinecone variables are missing."""

    validate_env(
        [
            "PINECONE_API_KEY",
            "PINECONE_INDEX_NAME",
            "PINECONE_CLOUD",
            "PINECONE_REGION",
        ]
    )


def _extract_index_names(list_indexes_response: Any) -> list[str]:
    """Normalize index-list response across Pinecone SDK versions."""

    if list_indexes_response is None:
        return []

    # Newer SDKs may expose `.names()` helper.
    if hasattr(list_indexes_response, "names"):
        names = list_indexes_response.names()
        return [str(name) for name in names]

    if isinstance(list_indexes_response, list):
        names: list[str] = []
        for item in list_indexes_response:
            if isinstance(item, dict) and "name" in item:
                names.append(str(item["name"]))
            else:
                names.append(str(item))
        return names

    # Fallback for typed response objects with `.indexes`.
    indexes = getattr(list_indexes_response, "indexes", None)
    if indexes:
        return [str(getattr(idx, "name", idx)) for idx in indexes]

    return []


def _extract_dimension(index_description: Any) -> int | None:
    """Read index dimension across SDK response shapes."""

    if index_description is None:
        return None

    for attr in ("dimension",):
        if hasattr(index_description, attr):
            value = getattr(index_description, attr)
            if isinstance(value, int):
                return value

    if isinstance(index_description, dict):
        if isinstance(index_description.get("dimension"), int):
            return int(index_description["dimension"])

        spec = index_description.get("spec")
        if isinstance(spec, dict):
            serverless = spec.get("serverless")
            if isinstance(serverless, dict) and isinstance(serverless.get("dimension"), int):
                return int(serverless["dimension"])

    return None


def get_or_create_index(expected_dimension: int) -> Any:
    """Return Pinecone index handle, creating index if missing.

    Parameters
    ----------
    expected_dimension:
        Embedding dimension expected by the selected embedding model.
    """

    ensure_pinecone_env()
    Pinecone, ServerlessSpec = _load_pinecone_sdk()

    client = Pinecone(api_key=settings.pinecone_api_key)
    existing_names = _extract_index_names(client.list_indexes())

    if settings.pinecone_index_name not in existing_names:
        logger.info(
            "Creating Pinecone index",
            extra={
                "context": {
                    "index": settings.pinecone_index_name,
                    "dimension": expected_dimension,
                    "cloud": settings.pinecone_cloud,
                    "region": settings.pinecone_region,
                }
            },
        )
        client.create_index(
            name=settings.pinecone_index_name,
            dimension=expected_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    else:
        description = client.describe_index(settings.pinecone_index_name)
        existing_dimension = _extract_dimension(description)
        if existing_dimension is not None and existing_dimension != expected_dimension:
            raise RuntimeError(
                "Pinecone index dimension mismatch: "
                f"index={existing_dimension}, embedder={expected_dimension}. "
                "Use a matching embedding model or recreate the index."
            )

    return client.Index(settings.pinecone_index_name)
