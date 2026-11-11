"""Small, non-authoritative names for already bound audio references.

Handles help a reader distinguish assets. They never replace canonical refs,
resolve natural-language intent, or authorize a different asset/version.
"""

from __future__ import annotations

import base64
import hashlib
import re


_SCENARIO_REF = re.compile(r"aud:v1:([sr])(\d{4,})")


def audio_reference_inventory(source_refs, reference_refs):
    refs = list(dict.fromkeys([*source_refs, *reference_refs]))
    digests = {
        ref: base64.b32encode(hashlib.sha256(ref.encode()).digest()).decode().lower()
        for ref in refs
    }
    rows = []
    for ref in refs:
        length = 3
        # Extend a colliding display name; never alias two assets to one name.
        while any(other != ref and value[:length] == digests[ref][:length]
                  for other, value in digests.items()):
            length += 1
        match = _SCENARIO_REF.fullmatch(ref)
        rows.append({
            "handle": digests[ref][:length],
            "audio_ref": ref,
            "attachment_position": int(match[2]) + 1 if match else None,
            "listed_as": [role for role, values in (
                ("source", source_refs), ("reference", reference_refs)) if ref in values],
        })
    return {
        "assets": rows,
        "policy": (
            "Handles are conversation-local display names, not speaker IDs or tool arguments. "
            "Use audio_ref verbatim in observations, evidence citations and PlanIR. "
            "attachment_position is the original supplied attachment order, not current list "
            "position or speaker order; null means no original attachment ordinal is asserted. "
            "Resolve the user's named/ordinal/semantic references against this mapping and "
            "audio-scoped evidence. Preserve the requested source, reference and version; "
            "listed_as does not decide a tool argument's edit/reference role. Never treat "
            "a previous output as the original recording. If a requested asset is absent, "
            "report it instead of remapping to another one."
        ),
    }
