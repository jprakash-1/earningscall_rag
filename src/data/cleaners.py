"""Text normalization helpers used during dataset ingestion.

Teaching goal:
Raw transcript text often carries inconsistent spacing, accidental control
characters, and uneven punctuation boundaries. This module applies conservative
cleaning so we preserve meaning while improving retrieval quality.
"""

from __future__ import annotations

import re
from typing import Any

# Regex to strip non-printable control characters while preserving tabs/newlines.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Collapse repeated whitespace but keep paragraph/newline boundaries readable.
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text(value: Any) -> str:
    """Normalize user or dataset text into a retrieval-friendly string.

    Parameters
    ----------
    value:
        Any incoming object from the dataset. We cast to string safely because
        upstream fields sometimes vary in type.

    Returns
    -------
    str
        A cleaned string with control characters removed, whitespace normalized,
        and outer boundaries trimmed.
    """

    if value is None:
        return ""

    text = str(value)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def normalize_metadata_value(value: Any) -> str | int | float | bool | None:
    """Normalize metadata to JSON-serializable scalar values.

    Why scalar normalization matters:
    Vector stores handle primitive metadata types reliably. Complex nested
    objects can be hard to filter on and may be rejected by some providers.
    """

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return normalize_text(value)
