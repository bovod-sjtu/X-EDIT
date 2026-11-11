"""Resolve diagnostic references against only the supplied immutable JSON.

Numeric bracket and dotted indices are equivalent. An inventory list may also
be addressed by its unique, explicitly declared audio_ref. No fuzzy names,
case folding, positional guesses or access to Python attributes is supported.
"""

import re
from typing import Any


def evidence_reference_exists(reference: str, namespace: dict[str, Any]) -> bool:
    if not isinstance(reference, str) or not reference or len(reference) > 2048:
        return False
    # Exact top-level IDs may themselves contain punctuation.
    if reference in namespace:
        return True
    if re.search(r"\]\s*[^.\[\]\s]", reference):
        return False
    normalized = re.sub(r"\[\s*(0|[1-9][0-9]*)\s*\]", r".\1", reference)
    if "[" in normalized or "]" in normalized:
        return False
    parts = normalized.split(".")
    if any(not part for part in parts) or len(parts) > 64:
        return False
    node: Any = namespace
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list):
            if part.isdecimal() and int(part) < len(node):
                node = node[int(part)]
            else:
                matches = [item for item in node
                           if isinstance(item, dict) and item.get("audio_ref") == part]
                if len(matches) != 1:
                    return False
                node = matches[0]
        else:
            return False
    return True
