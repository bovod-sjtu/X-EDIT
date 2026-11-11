"""Source rendering for validated edit labels, independent of runtime or HTTP."""

from __future__ import annotations

import re
from html import unescape


def render_tagged_source(
    instruction: str, *, lexical: bool = False, decode_entities: bool = True
) -> str:
    """Render source words; inserted words are absent from lexical sources.

    The caller owns label validation. Emotion/prosody services retain literal
    text entities, while the lexical service decodes its XML text nodes.
    No casing or whitespace is changed here.
    """

    if lexical:
        instruction = re.sub(r"<ins\b[^>]*>.*?</ins>", "", instruction, flags=re.DOTALL)
    source = re.sub(r"<[^>]+>", "", instruction)
    return unescape(source) if decode_entities else source
