"""Stable ID helpers for deterministic document and chunk identifiers.

Why deterministic IDs:
- Re-indexing should overwrite the same vectors, not duplicate them.
- Tests become reproducible.
- We can trace chunks back to source records consistently.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(payload: Any, *, length: int = 16) -> str:
    """Return a deterministic short hash for any JSON-serializable payload.

    Parameters
    ----------
    payload:
        Arbitrary data that can be encoded to JSON.
    length:
        Number of leading hex characters to keep for readability.
    """

    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]


def build_doc_id(*parts: Any, prefix: str = "doc") -> str:
    """Build a stable document ID from ordered components.

    Example:
    `build_doc_id(company, quarter, transcript_text)` -> `doc_ab12cd...`
    """

    return f"{prefix}_{stable_hash(parts)}"
