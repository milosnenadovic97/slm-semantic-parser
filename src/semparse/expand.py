"""Expand energy-sector abbreviations (ASHP, GSHP, ...) before prompting.

The few-shot examples use full names, so the model does better on expanded text.
"""

from __future__ import annotations

import re

ABBREVIATIONS: dict[str, str] = {
    "ashp": "air source heat pump",
    "gshp": "ground source heat pump",
    "wshp": "water source heat pump",
    "wwhp": "water to water heat pump",
    "awhp": "air to water heat pump",
}

# Drop wrapping parens too: "(air source heat pump)" reads like "(control)".
_PATTERN = re.compile(
    r"\(?\b(" + "|".join(re.escape(k) for k in ABBREVIATIONS) + r")\b\)?",
    re.IGNORECASE,
)


def expand(text: str) -> str:
    """Expand known abbreviations in text (whole words, case-insensitive)."""
    return _PATTERN.sub(lambda m: ABBREVIATIONS[m.group(1).lower()], text)
