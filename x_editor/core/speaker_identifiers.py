"""Opaque speaker names for a single reviewed evidence namespace.

These labels identify observations, not the order in which people speak. Raw
diarizer IDs remain in the original report; projection maintains the mapping.
"""

from __future__ import annotations

import hashlib
import string
from collections.abc import Iterable

_ALPHABET = string.ascii_lowercase
_CAPACITY = len(_ALPHABET) ** 3


def allocate_speaker_codes(
    local_ids: Iterable[str], *, namespace: str, reserved: Iterable[str] = ()
) -> dict[str, str]:
    """Allocate reproducible three-letter codes, resolving collisions explicitly.

    Sorting removes dependence on report row order. Allocation is scoped to the
    complete report, rather than claiming cross-job speaker identity without
    acoustic identity evidence. Already projected codes may be reserved.
    """

    identifiers = sorted(set(local_ids))
    used = set(reserved)
    if len(identifiers) + len(used) > _CAPACITY:
        raise ValueError("three-character speaker namespace capacity exceeded")
    result: dict[str, str] = {}
    for local_id in identifiers:
        digest = hashlib.sha256(f"{namespace}\0{local_id}".encode()).digest()
        start = int.from_bytes(digest[:8], "big") % _CAPACITY
        for offset in range(_CAPACITY):
            value = (start + offset) % _CAPACITY
            code = "".join(
                _ALPHABET[(value // (len(_ALPHABET) ** power)) % len(_ALPHABET)]
                for power in (2, 1, 0)
            )
            if code not in used:
                result[local_id] = code
                used.add(code)
                break
        else:
            raise ValueError("three-character speaker namespace exhausted")
    return result


def is_identity_label(label: str, local_id: str) -> bool:
    """Recognize formatting variants of an ID, without interpreting descriptions."""

    def compact(value: str) -> str:
        return "".join(character for character in value.casefold() if character.isalnum())

    label_key, identity_key = compact(label), compact(local_id)
    return label_key == identity_key or (
        local_id.isdecimal() and label_key == f"speaker{identity_key}"
    )
