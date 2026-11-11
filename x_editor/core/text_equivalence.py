"""Comparison of redundant natural-language text at component boundaries."""

from __future__ import annotations

import unicodedata


def equivalent_text_copy(left: str, right: str) -> bool:
    """Ignore casing, whitespace runs, and canonical Unicode representation.

    This is a comparison only: callers retain both original strings. It is
    not semantic similarity and must not be used for identifiers, paths,
    exact substring locations, or edits where capitalization is the goal.
    Punctuation, numbers, word order, and diacritics remain significant.
    """

    def key(value: str) -> str:
        return " ".join(unicodedata.normalize("NFC", value).casefold().split())

    return key(left) == key(right)


TRANSCRIPT_COPY_EQUIVALENCE = "unicode_spoken_tokens_v1"


def equivalent_transcript_copy(left: str, right: str) -> bool:
    """Compare redundant transcripts, tolerating sentence punctuation.

    This is lexical equality, not semantic similarity or a target locator.
    Sentence punctuation and casing can differ between ASR and a tagged copy.
    Word boundaries, order, repetitions, diacritics, numbers, contractions,
    and mathematical symbols remain significant. The authored strings and
    edit tags are never rewritten. Requests/identifiers must not use this
    comparison: punctuation may itself be their requested edit or value.
    """
    if equivalent_text_copy(left, right):
        return True
    tokens = _transcript_tokens(left)
    return bool(tokens) and tokens == _transcript_tokens(right)


def _transcript_tokens(value: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFC", value).casefold()
    # Width variants of punctuation use the same contract, without folding
    # compatibility characters in words/numbers (for example, superscripts).
    text = "".join(
        unicodedata.normalize("NFKC", char)
        if unicodedata.category(char).startswith("P") else char
        for char in text
    )
    tokens: list[str] = []
    word: list[str] = []

    def flush() -> None:
        if word:
            tokens.append("".join(word))
            word.clear()

    def lexical(char: str) -> bool:
        return bool(char) and unicodedata.category(char)[0] in "LNM"

    for index, char in enumerate(text):
        category = unicodedata.category(char)
        before = text[index - 1] if index else ""
        after = text[index + 1] if index + 1 < len(text) else ""
        if char in "'’ʼ" and lexical(before) and lexical(after):
            word.append("'")
        elif lexical(char):
            # Unspaced East Asian writing must not acquire artificial word
            # boundaries when a comma or full stop is added/removed.
            if category == "Lo" and unicodedata.east_asian_width(char) in "WF":
                flush()
                tokens.append(char)
            else:
                word.append(char)
        elif char in ".,:" and before.isdecimal() and after.isdecimal():
            word.append(char)
        elif char == "." and lexical(before) and lexical(after):
            # Retain abbreviations and dotted names; do not turn U.S into US.
            word.append(char)
        else:
            flush()
            if category[0] == "S" or category == "Pc" or char in "-‐‑/%&@#*\\":
                tokens.append(char)
            elif (char == "!" and before.isdecimal()) or (
                category == "Pd" and (before.isdecimal() or after.isdecimal())
            ):
                tokens.append(char)
            elif not char.isspace() and category[0] != "P":
                # Unknown/control characters are evidence, not free deletions.
                tokens.append(char)
    flush()
    return tuple(tokens)
